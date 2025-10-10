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

 
class RerankDataset(Dataset):
    """数据集类，用于组织query和候选图片对"""
    
    def __init__(self, json_data, image_dir):
        self.image_dir = image_dir
        self.samples = []
        
        # 构建所有需要重排序的样本
        for query in json_data['queries']:
            query_id = query['query_id']
            reference_image = query['reference_image']
            modification_text = query['modification_text']
            
            # 对每个候选图片创建一个样本
            for result in query['retrieval_results']:
                sample = {
                    'query_id': query_id,
                    'reference_image': reference_image,
                    'modification_text': modification_text,
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


def load_image(image_path, max_size=1024):
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


def create_messages(reference_image, candidate_image, modification_text):
    """创建Qwen2VL的输入消息格式"""
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
Instruction:{modification_text}
Decide if the candidate image matches the result of applying the instruction to the reference image.
Return yes if all required elements implied by the instruction are satisfied (like counts, categories, attributes, spatial relations). If any required element is missing or contradicted, answer no.
Answer:"""
                }
            ]
        }
    ]
    return messages


def process_single_sample(sample, processor, image_dir, max_image_size=1024):
    """处理单个样本，返回输入和信息"""
    # 加载图片（限制尺寸）
    ref_image_path = os.path.join(image_dir, f"{sample['reference_image']}.png")
    cand_image_path = os.path.join(image_dir, f"{sample['candidate_image']}.png")
    
    ref_image = load_image(ref_image_path, max_size=max_image_size)
    cand_image = load_image(cand_image_path, max_size=max_image_size)
    
    # 创建消息
    messages = create_messages(ref_image, cand_image, sample['modification_text'])
    
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
    
    info = {
        'query_id': sample['query_id'],
        'candidate_image': sample['candidate_image'],
        'rank': sample['rank'],
        'original_score': sample['original_score'],
        'candidate_index': sample['candidate_index']
    }
    
    return inputs, info


def collate_fn_optimized(batch, processor, image_dir, max_image_size=1024):
    """优化的collate函数，逐个处理样本以减少内存峰值"""
    all_inputs = []
    batch_info = []
    
    for sample in batch:
        inputs, info = process_single_sample(sample, processor, image_dir, max_image_size)
        all_inputs.append(inputs)
        batch_info.append(info)
    
    # 批量padding（更高效的内存使用）
    # 找出最大长度
    max_length = max(inp['input_ids'].shape[1] for inp in all_inputs)
    
    # 手动padding到相同长度
    padded_input_ids = []
    padded_attention_mask = []
    padded_pixel_values = []
    padded_image_grid_thw = []
    
    for inp in all_inputs:
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
        yes_logits_cpu = yes_logits.cpu().numpy()
        no_logits_cpu = no_logits.cpu().numpy()
        
        # 清理中间变量
        del outputs, logits
        torch.cuda.empty_cache()
    
    return yes_probs, yes_logits_cpu, no_logits_cpu

def rerank_queries(all_results):
    """对每个query的候选结果进行重排序"""
    # 按query_id组织结果
    query_results = {}
    for result in all_results:
        query_id = result['query_id']
        if query_id not in query_results:
            query_results[query_id] = []
        query_results[query_id].append(result)
    
    # 对每个query进行重排序
    reranked_results = {}
    for query_id, candidates in query_results.items():
        # 按新分数排序
        sorted_candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # 更新排名
        for new_rank, candidate in enumerate(sorted_candidates, 1):
            candidate['new_rank'] = new_rank
        
        reranked_results[query_id] = sorted_candidates
    
    return reranked_results


def calculate_original_metrics(original_json):
    """计算原始检索结果的指标，包括R1、R5、R10"""
    found_at_1 = 0
    found_at_5 = 0
    found_at_10 = 0
    total_queries = len(original_json['queries'])
    
    for query in original_json['queries']:
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


def calculate_metrics(reranked_results, original_json):
    """计算重排序后的指标，包括R1、R5、R10"""
    found_at_1 = 0
    found_at_5 = 0
    found_at_10 = 0
    total_queries = len(original_json['queries'])
    
    for query in original_json['queries']:
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
    parser = argparse.ArgumentParser(description='Rerank retrieval results using Qwen2VL')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to Qwen2VL model directory')
    parser.add_argument('--json_file', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--output_file', type=str, default='reranked_results.json', help='Output JSON file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing (reduced default)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--max_image_size', type=int, default=1024, help='Maximum image dimension')
    
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
    dataset = RerankDataset(json_data, args.image_dir)
    print('num of triple samples: ', len(dataset))

    # 创建DataLoader时使用优化的collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn_optimized(
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
            # 计算yes/no分数和logit值
            yes_scores, yes_logits, no_logits = compute_yes_no_scores(model, batch_inputs, processor, device)
            
            # 组合结果
            for i, score in enumerate(yes_scores):
                result = batch_info[i].copy()
                result['rerank_score'] = float(score)
                result['yes_logit'] = float(yes_logits[i])
                result['no_logit'] = float(no_logits[i])
                all_results.append(result)
            
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
        print("Reranking results...")
        
        # 计算原始检索结果的指标
        original_metrics = calculate_original_metrics(json_data)
        
        # 重排序
        reranked_results = rerank_queries(all_results)
        
        # 计算重排序后的指标
        reranked_metrics = calculate_metrics(reranked_results, json_data)
        
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
        output_data = {
            'metadata': {
                **json_data['metadata'],
                'reranked': True,
                'rerank_model': args.model_dir,
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
        for query in json_data['queries']:
            query_id = query['query_id']
            query_output = query.copy()
            
            if query_id in reranked_results:
                query_output['reranked_results'] = []
                for candidate in reranked_results[query_id][:10]:  # 只保存top10
                    query_output['reranked_results'].append({
                        'rank': candidate['new_rank'],
                        'candidate_image': candidate['candidate_image'],
                        'rerank_score': candidate['rerank_score'],
                        'yes_logit': candidate['yes_logit'],
                        'no_logit': candidate['no_logit'],
                        'original_rank': candidate['rank'],
                        'original_score': candidate['original_score']
                    })
            
            output_data['queries'].append(query_output)
        
        # 保存到文件
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()

'''
运行命令（按稳定性从高到低）:

# 最稳定：单样本处理，不使用Flash Attention
accelerate launch --num_processes 8 VQA_rerank_single.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --json_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/retrieval_results/checkpoint-1500_20250919_160740/cirr_retrieval_top10.json \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/qwen2_7b/R1_49/reranked_results_single.json \
    --batch_size 2 \
    --max_image_size 2048

# 较快：批处理，使用Flash Attention
accelerate launch --num_processes 8 VQA_rerank_single.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --json_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/retrieval_results/base_model_20250910_180548/cirr_retrieval_top10.json \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file reranked_results.json \
    --batch_size 2 \
    --max_image_size 768 \
    --use_flash_attention

# 如果内存充足：更大的批处理
accelerate launch --num_processes 8 VQA_rerank_single.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --json_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/retrieval_results/base_model_20250910_180548/cirr_retrieval_top10.json \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file reranked_results.json \
    --batch_size 4 \
    --max_image_size 512
'''
"""prompt1:
You are a strict visual verifier. Output exactly one token: yes or no (lowercase).Do not add punctuation or explanations.
Reference image: Picture1
Candidate image: Picture2
Instruction:{modification_text}
Decide if the candidate image matches the result of applying the instruction to the reference image.
Return yes if all required elements implied by the instruction are satisfied (like counts, categories, attributes, spatial relations). If any required element is missing or contradicted, answer no.
Answer:"""