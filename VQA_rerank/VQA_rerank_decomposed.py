import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from accelerate.utils import gather_object
import numpy as np
from typing import List, Dict, Any
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings('ignore')

 
class DecomposedRerankDataset(Dataset):
    """支持子修改文本的数据集类，用于组织query和候选图片对"""
    
    def __init__(self, json_data, image_dir, max_queries=None):
        self.image_dir = image_dir
        self.samples = []
        
        # 限制查询数量（如果指定）
        queries_to_process = json_data['queries'][:max_queries] if max_queries else json_data['queries']
        
        # 构建所有需要重排序的样本
        for query in queries_to_process:
            query_id = query['query_id']
            reference_image = query['reference_image']
            modification_text = query['modification_text']
            sub_modifications = query.get('sub_modifications', [modification_text])  # 如果没有子修改，使用原修改文本
            
            # 对每个候选图片创建一个样本
            for result in query['retrieval_results']:
                sample = {
                    'query_id': query_id,
                    'reference_image': reference_image,
                    'modification_text': modification_text,
                    'sub_modifications': sub_modifications,  # 新增子修改文本列表
                    'candidate_image': result['candidate_image'],
                    'rank': result['rank'],
                    'original_score': result['similarity_score'],
                    'candidate_index': result['candidate_index']
                }
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def load_image(image_path, max_size=2048):
    """加载图片并限制最大尺寸，处理错误"""
    try:
        image = Image.open(image_path).convert('RGB')
        
        # 限制图像最大尺寸以节省内存
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # 返回一个小的空白图片作为fallback
        return Image.new('RGB', (224, 224), color='white')


def create_messages(reference_image, candidate_image, sub_modification_text):
    """创建Qwen2VL的输入消息格式，使用单个子修改文本"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": reference_image,
                },
                {
                    "type": "image", 
                    "image": candidate_image,
                },
                {
                    "type": "text",
                    "text": f"""You are a strict visual verifier. Output exactly one token: yes or no (lowercase).Do not add punctuation or explanations.
Reference image: Picture1
Candidate image: Picture2
Instruction:{sub_modification_text}
Decide if the candidate image matches the result of applying the instruction to the reference image.
Return yes if all required elements implied by the instruction are satisfied (like counts, categories, attributes, spatial relations). If any required element is missing or contradicted, answer no.
Answer:"""
                }
            ]
        }
    ]
    return messages


def process_sample_with_sub_modifications(sample, processor, image_dir, max_image_size=2048):
    """处理单个样本的所有子修改文本，返回所有输入和信息"""
    # 加载图片（限制尺寸）
    ref_image_path = os.path.join(image_dir, f"{sample['reference_image']}.png")
    cand_image_path = os.path.join(image_dir, f"{sample['candidate_image']}.png")
    
    ref_image = load_image(ref_image_path, max_size=max_image_size)
    cand_image = load_image(cand_image_path, max_size=max_image_size)
    
    all_inputs = []
    sub_mod_texts = []
    
    # 为每个子修改文本创建输入
    for sub_modification in sample['sub_modifications']:
        # 创建消息
        messages = create_messages(ref_image, cand_image, sub_modification)
        
        # 处理单个样本
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            add_vision_id=True,
        )
        
        # 处理图像信息
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 确保使用与tokenizer.padding_side一致的padding设置
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=False,  # 不在这里padding
            return_tensors="pt"
        )
        
        all_inputs.append(inputs)
        sub_mod_texts.append(sub_modification)
    
    info = {
        'query_id': sample['query_id'],
        'candidate_image': sample['candidate_image'],
        'rank': sample['rank'],
        'original_score': sample['original_score'],
        'candidate_index': sample['candidate_index'],
        'sub_modifications': sub_mod_texts,
        'num_sub_modifications': len(sub_mod_texts)
    }
    
    return all_inputs, info


def collate_fn_decomposed(batch, processor, image_dir, max_image_size=1024):
    """优化的collate函数，处理包含子修改的批次"""
    all_sub_inputs = []
    batch_info = []
    
    for sample in batch:
        sub_inputs_list, info = process_sample_with_sub_modifications(
            sample, processor, image_dir, max_image_size
        )
        all_sub_inputs.extend(sub_inputs_list)  # 展开所有子修改的输入
        batch_info.append(info)
    
    # 如果没有输入，返回空
    if not all_sub_inputs:
        return {}, batch_info
    
    # 批量padding（更高效的内存使用）
    # 找出最大长度
    max_length = max(inp['input_ids'].shape[1] for inp in all_sub_inputs)
    
    # 手动padding到相同长度
    padded_input_ids = []
    padded_attention_mask = []
    padded_pixel_values = []
    padded_image_grid_thw = []
    
    for inp in all_sub_inputs:
        # Padding input_ids
        input_ids = inp['input_ids']
        current_length = input_ids.shape[1]
        if current_length < max_length:
            # 根据padding_side决定padding的位置
            if processor.tokenizer.padding_side == 'left':
                # 左侧padding
                padding = torch.full((1, max_length - current_length), processor.tokenizer.pad_token_id)
                input_ids = torch.cat([padding, input_ids], dim=1)
            else:
                # 右侧padding
                padding = torch.full((1, max_length - current_length), processor.tokenizer.pad_token_id)
                input_ids = torch.cat([input_ids, padding], dim=1)
        padded_input_ids.append(input_ids)
        
        # Padding attention_mask
        attention_mask = inp['attention_mask']
        if attention_mask.shape[1] < max_length:
            # 根据padding_side决定padding的位置
            if processor.tokenizer.padding_side == 'left':
                # 左侧padding (0)
                padding = torch.zeros((1, max_length - attention_mask.shape[1]))
                attention_mask = torch.cat([padding, attention_mask], dim=1)
            else:
                # 右侧padding (0)
                padding = torch.zeros((1, max_length - attention_mask.shape[1]))
                attention_mask = torch.cat([attention_mask, padding], dim=1)
        padded_attention_mask.append(attention_mask)
        
        # 处理图像相关的张量
        if 'pixel_values' in inp:
            padded_pixel_values.append(inp['pixel_values'])
        if 'image_grid_thw' in inp:
            padded_image_grid_thw.append(inp['image_grid_thw'])
    
    # 组合batch
    batch_inputs = {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'attention_mask': torch.cat(padded_attention_mask, dim=0)
    }
    
    if padded_pixel_values:
        batch_inputs['pixel_values'] = torch.cat(padded_pixel_values, dim=0)
    if padded_image_grid_thw:
        batch_inputs['image_grid_thw'] = torch.cat(padded_image_grid_thw, dim=0)
    
    return batch_inputs, batch_info


def compute_yes_no_scores(model, inputs, processor, device):
    """计算yes/no的logits并转换为概率分数，同时返回原始logit值"""
    
    # 获取yes和no的token id
    yes_token_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_token_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]
    
    # 将输入移到设备上
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        # 清理缓存
        torch.cuda.empty_cache()
        
        # 前向传播获取logits
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # 获取最后一个token的logits
        
        # 提取yes和no的logits
        yes_logits = logits[:, yes_token_id]
        no_logits = logits[:, no_token_id]
        
        # 组合并计算softmax
        yes_no_logits = torch.stack([no_logits, yes_logits], dim=1)
        probs = F.softmax(yes_no_logits, dim=1)
        
        # yes的概率作为相似度分数
        yes_probs = probs[:, 1].cpu().numpy()
        
        # 保存原始logit值
        yes_logits_values = yes_logits.cpu().numpy()
        no_logits_values = no_logits.cpu().numpy()
        
        # 清理中间变量
        del outputs, logits
        torch.cuda.empty_cache()
    
    # 返回yes概率和原始logit值
    return yes_probs, yes_logits_values, no_logits_values


def process_decomposed_scores(all_sub_data, batch_info, aggregation_mode="arithmetic"):
    """将所有子修改的得分重新组织并聚合
    
    Args:
        all_sub_data: 包含所有子修改得分和logit值的元组 (yes_probs, yes_logits, no_logits)
        batch_info: 每个样本的信息
        aggregation_mode: 聚合模式，可选值有:
            - "arithmetic": 算术平均值 (默认)
            - "geometric": 几何平均值
            - "harmonic": 调和平均值
            - "median": 中位数
            - "min": 最小值
            - "max": 最大值
    """
    results = []
    score_idx = 0
    
    # 解包数据
    all_sub_scores, all_yes_logits, all_no_logits = all_sub_data
    
    for info in batch_info:
        num_sub_mods = info['num_sub_modifications']
        
        # 提取当前样本的所有子修改得分和logit值
        sub_scores = all_sub_scores[score_idx:score_idx + num_sub_mods]
        sub_yes_logits = all_yes_logits[score_idx:score_idx + num_sub_mods]
        sub_no_logits = all_no_logits[score_idx:score_idx + num_sub_mods]
        score_idx += num_sub_mods
        
        # 根据指定的聚合模式计算聚合得分
        if aggregation_mode == "arithmetic":
            # 算术平均值
            aggregated_score = np.mean(sub_scores)
        elif aggregation_mode == "geometric":
            # 几何平均值
            aggregated_score = np.exp(np.mean(np.log(np.maximum(sub_scores, 1e-10))))
        elif aggregation_mode == "harmonic":
            # 调和平均值
            aggregated_score = len(sub_scores) / np.sum(1.0 / np.maximum(sub_scores, 1e-10))
        elif aggregation_mode == "median":
            # 中位数
            aggregated_score = np.median(sub_scores)
        elif aggregation_mode == "min":
            # 最小值
            aggregated_score = np.min(sub_scores)
        elif aggregation_mode == "max":
            # 最大值
            aggregated_score = np.max(sub_scores)
        else:
            # 默认使用算术平均值
            print(f"警告: 未知的聚合模式 '{aggregation_mode}'，使用算术平均值代替")
            aggregated_score = np.mean(sub_scores)
        
        # 构建结果
        result = {
            'query_id': info['query_id'],
            'candidate_image': info['candidate_image'],
            'rank': info['rank'],
            'original_score': info['original_score'],
            'candidate_index': info['candidate_index'],
            'sub_modification_scores': {
                text: {
                    'score': float(score),
                    'yes_logit': float(yes_logit),
                    'no_logit': float(no_logit)
                } for text, score, yes_logit, no_logit in zip(info['sub_modifications'], sub_scores, sub_yes_logits, sub_no_logits)
            },
            'aggregation_mode': aggregation_mode,
            'aggregated_score': float(aggregated_score),
            'rerank_score': float(aggregated_score)  # 使用聚合得分作为重排分数
        }
        results.append(result)
    
    return results


def rerank_queries(all_results):
    """对每个query的候选结果进行重排序，并去除重复项"""
    # 按query_id组织结果，同时去重
    query_results = {}
    for result in all_results:
        query_id = result['query_id']
        candidate_image = result['candidate_image']
        
        if query_id not in query_results:
            query_results[query_id] = {}
        
        # 使用candidate_image作为key来去重，如果已存在则跳过
        # 这样可以避免多进程处理时产生的重复结果
        if candidate_image not in query_results[query_id]:
            query_results[query_id][candidate_image] = result
        else:
            # 如果出现重复，选择保留分数更高的那个
            existing_score = query_results[query_id][candidate_image]['rerank_score']
            new_score = result['rerank_score']
            if new_score > existing_score:
                query_results[query_id][candidate_image] = result
    
    # 对每个query进行重排序
    reranked_results = {}
    for query_id, candidates_dict in query_results.items():
        # 将字典转换为列表
        candidates = list(candidates_dict.values())
        
        # 按新分数排序
        sorted_candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # 更新排名
        for new_rank, candidate in enumerate(sorted_candidates, 1):
            candidate['new_rank'] = new_rank
        
        reranked_results[query_id] = sorted_candidates
    
    return reranked_results


def calculate_original_metrics(original_json, max_queries=None):
    """计算原始检索结果的指标，包括R1、R5、R10"""
    found_at_1 = 0
    found_at_5 = 0
    found_at_10 = 0
    
    queries_to_process = original_json['queries'][:max_queries] if max_queries else original_json['queries']
    total_queries = len(queries_to_process)
    
    for query in queries_to_process:
        target_hard = query['target_hard']
        results = query['retrieval_results']
        
        # 检查target是否在top1中
        if results[0]['candidate_image'] == target_hard:
            found_at_1 += 1
            found_at_5 += 1
            found_at_10 += 1
        # 检查target是否在top5中
        elif any(result['candidate_image'] == target_hard for result in results[:5]):
            found_at_5 += 1
            found_at_10 += 1
        # 检查target是否在top10中
        elif any(result['candidate_image'] == target_hard for result in results[:10]):
            found_at_10 += 1
    
    r1 = found_at_1 / total_queries if total_queries > 0 else 0
    r5 = found_at_5 / total_queries if total_queries > 0 else 0
    r10 = found_at_10 / total_queries if total_queries > 0 else 0
    
    return {
        'total_queries': total_queries,
        'found_at_1': found_at_1,
        'found_at_5': found_at_5,
        'found_at_10': found_at_10,
        'r1': r1,
        'r5': r5,
        'r10': r10
    }


def calculate_metrics(reranked_results, original_json, max_queries=None):
    """计算重排序后的指标，包括R1、R5、R10"""
    found_at_1 = 0
    found_at_5 = 0
    found_at_10 = 0
    
    queries_to_process = original_json['queries'][:max_queries] if max_queries else original_json['queries']
    total_queries = len(queries_to_process)
    
    for query in queries_to_process:
        query_id = query['query_id']
        target_hard = query['target_hard']
        
        if query_id in reranked_results:
            # 检查target是否在top1中
            if reranked_results[query_id][0]['candidate_image'] == target_hard:
                found_at_1 += 1
                found_at_5 += 1
                found_at_10 += 1
            # 检查target是否在top5中
            elif any(candidate['candidate_image'] == target_hard for candidate in reranked_results[query_id][:5]):
                found_at_5 += 1
                found_at_10 += 1
            # 检查target是否在top10中
            elif any(candidate['candidate_image'] == target_hard for candidate in reranked_results[query_id][:10]):
                found_at_10 += 1
    
    r1 = found_at_1 / total_queries if total_queries > 0 else 0
    r5 = found_at_5 / total_queries if total_queries > 0 else 0
    r10 = found_at_10 / total_queries if total_queries > 0 else 0
    
    return {
        'total_queries': total_queries,
        'found_at_1': found_at_1,
        'found_at_5': found_at_5,
        'found_at_10': found_at_10,
        'r1': r1,
        'r5': r5,
        'r10': r10
    }


def main():
    parser = argparse.ArgumentParser(description='Rerank retrieval results using Qwen2VL with decomposed modifications')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to Qwen2VL model directory')
    parser.add_argument('--json_file', type=str, 
                       default='/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/R1_49_2_5_7b_decomposed_results_2.json', 
                       help='Path to input JSON file with decomposed modifications')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--output_file', type=str, default='decomposed_reranked_results.json', help='Output JSON file')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for processing (reduced for sub-modifications)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--max_image_size', type=int, default=1024, help='Maximum image dimension')
    parser.add_argument('--max_queries', type=int, default=None, help='Maximum number of queries to process (for testing)')
    parser.add_argument('--aggregation_mode', type=str, default='arithmetic', 
                      choices=['arithmetic', 'geometric', 'harmonic', 'median', 'min', 'max'],
                      help='子修改得分聚合方式: arithmetic(算术平均值), geometric(几何平均值), harmonic(调和平均值), median(中位数), min(最小值), max(最大值)')
    
    args = parser.parse_args()
    
    # 设置环境变量以优化CUDA内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 初始化accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # 仅在主进程中打印信息
    if accelerator.is_main_process:
        print(f"Loading model from {args.model_dir}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Device: {device}")
        print(f"Max image size: {args.max_image_size}")
        print(f"Batch size: {args.batch_size}")
        if args.max_queries:
            print(f"Max queries to process: {args.max_queries}")
    
    # 加载processor
    processor = AutoProcessor.from_pretrained(args.model_dir,min_pixels=4*28*28,max_pixels=2048*28*28)
    
    # 设置tokenizer的padding_side为'left'，以兼容Flash Attention
    # 必须在创建数据加载器之前设置
    processor.tokenizer.padding_side = 'left'
    
    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map={"": device},
        attn_implementation="flash_attention_2"  # 使用Flash Attention节省内存
    )
    
    # 设置模型为评估模式
    model.eval()
    
    # 加载JSON数据
    if accelerator.is_main_process:
        print(f"Loading data from {args.json_file}")
    
    with open(args.json_file, 'r') as f:
        json_data = json.load(f)
    
    # 创建数据集和数据加载器
    dataset = DecomposedRerankDataset(json_data, args.image_dir, max_queries=args.max_queries)
    
    if accelerator.is_main_process:
        print(f'Number of candidate samples: {len(dataset)}')
        if args.max_queries:
            print(f'Processing first {args.max_queries} queries')

    # 创建DataLoader时使用优化的collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn_decomposed(
            batch, processor, args.image_dir, args.max_image_size
        ),
        pin_memory=False  # 避免额外的内存占用
    )
    
    # 使用accelerator准备dataloader
    dataloader = accelerator.prepare(dataloader)
    
    # 处理所有批次
    all_results = []
    
    if accelerator.is_main_process:
        pbar = tqdm(total=len(dataloader), desc="Processing batches")
    
    for batch_inputs, batch_info in dataloader:
        try:
            # 如果没有有效输入，跳过
            if not batch_inputs:
                if accelerator.is_main_process:
                    pbar.update(1)
                continue
                
            # 计算yes/no分数及logit值
            all_sub_scores, all_yes_logits, all_no_logits = compute_yes_no_scores(model, batch_inputs, processor, device)
            
            # 处理分解的得分并根据指定聚合模式计算
            batch_results = process_decomposed_scores((all_sub_scores, all_yes_logits, all_no_logits), batch_info, args.aggregation_mode)
            all_results.extend(batch_results)
            
            if accelerator.is_main_process:
                pbar.update(1)
                
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM error in batch, skipping: {e}")
            torch.cuda.empty_cache()
            continue
    
    if accelerator.is_main_process:
        pbar.close()
    
    # 收集所有进程的结果
    all_results = gather_object(all_results)
    
    # 仅在主进程中进行重排序和保存结果
    if accelerator.is_main_process:
        # 添加调试信息
        print(f"Total results collected from all processes: {len(all_results)}")
        
        # 统计每个query的结果数量，检查是否有重复
        query_counts = {}
        for result in all_results:
            query_id = result['query_id']
            candidate_image = result['candidate_image']
            if query_id not in query_counts:
                query_counts[query_id] = {}
            if candidate_image not in query_counts[query_id]:
                query_counts[query_id][candidate_image] = 0
            query_counts[query_id][candidate_image] += 1
        
        # 检查是否有重复并报告
        total_duplicates = 0
        for query_id, candidates in query_counts.items():
            for candidate_image, count in candidates.items():
                if count > 1:
                    total_duplicates += count - 1
        
        if total_duplicates > 0:
            print(f"Warning: Found {total_duplicates} duplicate results, will be automatically removed")
        
        print("Reranking results...")
        
        # 计算原始检索结果的指标
        original_metrics = calculate_original_metrics(json_data, args.max_queries)
        
        # 重排序
        reranked_results = rerank_queries(all_results)
        
        # 计算重排序后的指标
        reranked_metrics = calculate_metrics(reranked_results, json_data, args.max_queries)
        
        # 打印原始指标
        print(f"\n原始检索结果指标:")
        print(f"Original R@1: {original_metrics['r1']:.4f} ({original_metrics['found_at_1']}/{original_metrics['total_queries']})")
        print(f"Original R@5: {original_metrics['r5']:.4f} ({original_metrics['found_at_5']}/{original_metrics['total_queries']})")
        print(f"Original R@10: {original_metrics['r10']:.4f} ({original_metrics['found_at_10']}/{original_metrics['total_queries']})")
        
        # 打印重排序后的指标
        print(f"\n重排序后指标:")
        print(f"Reranked R@1: {reranked_metrics['r1']:.4f} ({reranked_metrics['found_at_1']}/{reranked_metrics['total_queries']})")
        print(f"Reranked R@5: {reranked_metrics['r5']:.4f} ({reranked_metrics['found_at_5']}/{reranked_metrics['total_queries']})")
        print(f"Reranked R@10: {reranked_metrics['r10']:.4f} ({reranked_metrics['found_at_10']}/{reranked_metrics['total_queries']})")
        
        # 保存结果
        queries_to_save = json_data['queries'][:args.max_queries] if args.max_queries else json_data['queries']
        
        output_data = {
            'metadata': {
                **json_data['metadata'],
                'reranked': True,
                'rerank_model': args.model_dir,
                'max_queries_processed': len(queries_to_save),
                'aggregation_mode': args.aggregation_mode,  # 添加聚合模式信息
                'original_r1': original_metrics['r1'],
                'original_r5': original_metrics['r5'],
                'original_r10': original_metrics['r10'],
                'original_found_at_1': original_metrics['found_at_1'],
                'original_found_at_5': original_metrics['found_at_5'],
                'original_found_at_10': original_metrics['found_at_10'],
                'rerank_r1': reranked_metrics['r1'],
                'rerank_r5': reranked_metrics['r5'],
                'rerank_r10': reranked_metrics['r10'],
                'rerank_found_at_1': reranked_metrics['found_at_1'],
                'rerank_found_at_5': reranked_metrics['found_at_5'],
                'rerank_found_at_10': reranked_metrics['found_at_10']
            },
            'queries': []
        }
        
        # 组织输出数据
        for query in queries_to_save:
            query_id = query['query_id']
            query_output = query.copy()
            
            if query_id in reranked_results:
                query_output['reranked_results'] = []
                for candidate in reranked_results[query_id][:10]:  # 只保存top10
                    result_output = {
                        'rank': candidate['new_rank'],
                        'candidate_image': candidate['candidate_image'],
                        'rerank_score': candidate['rerank_score'],
                        'aggregated_score': candidate['aggregated_score'],
                        'aggregation_mode': candidate['aggregation_mode'],
                        'original_rank': candidate['rank'],
                        'original_score': candidate['original_score'],
                        'sub_modification_scores': candidate['sub_modification_scores']  # 各子修改的详细得分
                    }
                    query_output['reranked_results'].append(result_output)
            
            output_data['queries'].append(query_output)
        
        # 保存到文件
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()

'''
运行命令示例:

# 处理所有query (默认使用算术平均值聚合)
accelerate launch --num_processes 8 VQA_rerank_decomposed.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file decomposed_reranked_results.json \
    --batch_size 2 \
    --max_image_size 768

# 使用几何平均值聚合
accelerate launch --num_processes 8 VQA_rerank_decomposed.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/R1_49/2048_2048*28*28_decomposed_reranked_results.json \
    --batch_size 2 \
    --max_image_size 2048 \
    --aggregation_mode geometric

# 使用调和平均值聚合
accelerate launch --num_processes 8 VQA_rerank_decomposed.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/R1_49/2048_2048*28*28_decomposed_reranked_results.json \
    --batch_size 2 \
    --max_image_size 2048 \
    --aggregation_mode harmonic

# 只处理前100个query进行测试
accelerate launch --num_processes 8 VQA_rerank_decomposed.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file decomposed_reranked_results_test.json \
    --batch_size 2 \
    --max_image_size 768 \
    --max_queries 100

# 只处理前50个query进行快速测试
accelerate launch --num_processes 8 VQA_rerank_decomposed.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/qwen2_7b/R1_49/decomposed_reranked_results_small_test2.json \
    --batch_size 2 \
    --max_image_size 1280 \
    --max_queries 50
    --aggregation_mode geometric

# 使用单进程版本并指定使用最小值聚合
python VQA_rerank_decomposed.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/decomposed_reranked_results_min.json \
    --batch_size 4 \
    --max_image_size 768 \
    --max_queries 50 \
    --aggregation_mode min
'''
