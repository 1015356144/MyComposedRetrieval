import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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


class TextDecompositionDataset(Dataset):
    """用于文本分解的数据集类"""
    
    def __init__(self, json_data):
        self.queries = json_data['queries']
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return self.queries[idx]


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


def create_decomposition_messages(modification_text):
    """创建用于文本分解的消息格式"""
    system_prompt="""
    You are a rigorous instruction decomposer for image edit verification. Your goal is to convert a single modification instruction into multiple atomic, independently-verifiable sub-modifications.

## Core rules
1) Granularity: Each sub-modification must express **exactly one** requirement.
2) Verifiability: Each sub-modification must be checkable from **images only** (no reasoning beyond what the text states).
3) No hallucination: **Do not** invent objects, attributes, relations, styles, counts, or text not explicitly required by the instruction.
4) Full sentence form: Each sub-modification MUST be a **complete English sentence** starting with:
   **"In the target image, ..."**
5) Relative vs absolute:
   - If the instruction implies a change from the reference to the target, use “is changed to / is replaced by / is removed / is added”.
   - If it requires an absolute property in the target, state it plainly (color, count, spatial relation, style, text content, etc.).
6) Split conjunctions: Break “and / or / with / while / then” into separate sentences whenever they encode different requirements.
7) No bundles: Avoid commas or “and” that combine multiple atomic facts in one sentence.
8) Keep uncertainty: If the phrase is ambiguous or underspecified, keep the exact requirement **as written** (do not expand it).

## Surface templates (use when applicable)
- Presence/absence: “In the target image, **<object> is present / is removed**.”
- Attribute change: “In the target image, **the <object> is changed to <color/size/material>**.”
- Category replacement: “In the target image, **the <object A> is replaced by a <object B>**.”
- Count: “In the target image, **there are <N> <objects>**.”
- Spatial relation: “In the target image, **the <object A> is <left of / on / in front of / overlapping> the <object B>**.”
- Action/state: “In the target image, **the <object> is <running / open / closed>**.”
- Text-in-image: “In the target image, **the text reads '<STRING>'**.”
- Style/global: “In the target image, **the photo has a <style> style**.”
- Camera/view: “In the target image, **the scene is shown from a <view>**.”

## Bad vs Good (micro-examples)
- BAD: “In the target image, the car is red and has two doors.”  (two facts bundled)
- GOOD: 
  1) “In the target image, the car is red.”
  2) “In the target image, the car has two doors.”

## In-context examples

### Example A
Instruction: “Change the blue car to a red sports car and move it to the left of the tree.”
Output:
{
  "sub_modifications": [
    "In the target image, the blue car is changed to a red car.",
    "In the target image, the car is a sports car.",
    "In the target image, the car is to the left of the tree."
  ]
}

### Example B
Instruction: “Replace the dog with a cat wearing a green collar and make it sit on the sofa.”
Output:
{
  "sub_modifications": [
    "In the target image, the dog is replaced by a cat.",
    "In the target image, the cat is wearing a green collar.",
    "In the target image, the cat is sitting on the sofa."
  ]
}

### Example C
Instruction: “Add two cups on the table and place the book under the lamp.”
Output:
{
  "sub_modifications": [
    "In the target image, two cups are added on the table.",
    "In the target image, the book is under the lamp."
  ]
}

### Example D
Instruction: “Change the text on the sign to ‘OPEN’ and switch the scene to a nighttime style.”
Output:
{
  "sub_modifications": [
    "In the target image, the text on the sign reads 'OPEN'.",
    "In the target image, the photo has a nighttime style."
  ]
}

### Example E
Instruction: “Make the person wear a red hat, turn them to face right, and blur the background.”
Output:
{
  "sub_modifications": [
    "In the target image, the person is wearing a red hat.",
    "In the target image, the person is facing right.",
    "In the target image, the background is blurred."
  ]
}

## Output format (MUST follow exactly)
Return ONLY the JSON object below and nothing else:
{
    "sub_modifications": [
        "sub-modification 1",
        "sub-modification 2",
        "sub-modification 3"
    ]
}

"""
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Now decompose the actual instruction.
                            Modification instruction: "{modification_text}"
                """
                }
            ]
        }
    ]
    return messages


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


def process_text_decomposition_batch(batch, processor):
    """处理文本分解批次"""
    all_inputs = []
    batch_info = []
    
    for query in batch:
        # 创建消息
        messages = create_decomposition_messages(query['modification_text'])
        
        # 处理文本
        text = processor.apply_chat_template(
            [messages], 
            tokenize=False, 
            add_generation_prompt=True,
            add_vision_id=True,
        )
        
        # 处理输入（纯文本，无图像）
        inputs = processor(
            text=text,
            images=None,
            videos=None,
            padding=False,
            return_tensors="pt"
        )
        
        all_inputs.append(inputs)
        batch_info.append({
            'query_id': query['query_id'],
            'original_modification_text': query['modification_text']
        })
    
    # 批量padding
    if len(all_inputs) > 0:
        max_length = max(inp['input_ids'].shape[1] for inp in all_inputs)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for inp in all_inputs:
            input_ids = inp['input_ids']
            current_length = input_ids.shape[1]
            if current_length < max_length:
                if processor.tokenizer.padding_side == 'left':
                    padding = torch.full((1, max_length - current_length), processor.tokenizer.pad_token_id)
                    input_ids = torch.cat([padding, input_ids], dim=1)
                else:
                    padding = torch.full((1, max_length - current_length), processor.tokenizer.pad_token_id)
                    input_ids = torch.cat([input_ids, padding], dim=1)
            padded_input_ids.append(input_ids)
            
            attention_mask = inp['attention_mask']
            if attention_mask.shape[1] < max_length:
                if processor.tokenizer.padding_side == 'left':
                    padding = torch.zeros((1, max_length - attention_mask.shape[1]))
                    attention_mask = torch.cat([padding, attention_mask], dim=1)
                else:
                    padding = torch.zeros((1, max_length - attention_mask.shape[1]))
                    attention_mask = torch.cat([attention_mask, padding], dim=1)
            padded_attention_mask.append(attention_mask)
        
        batch_inputs = {
            'input_ids': torch.cat(padded_input_ids, dim=0),
            'attention_mask': torch.cat(padded_attention_mask, dim=0)
        }
    else:
        batch_inputs = {'input_ids': torch.empty(0), 'attention_mask': torch.empty(0)}
    
    return batch_inputs, batch_info


def generate_text_decomposition(model, inputs, processor, device, max_new_tokens=200):
    """生成文本分解结果"""
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        # 生成文本
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        
        # 解码生成的文本
        generated_texts = []
        for i, output in enumerate(outputs):
            # 只取新生成的部分
            generated_part = output[inputs['input_ids'][i].shape[0]:]
            generated_text = processor.tokenizer.decode(generated_part, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())
        
        torch.cuda.empty_cache()
    
    return generated_texts


def parse_decomposition_result(generated_text):
    """解析模型生成的分解结果"""
    try:
        # 尝试解析JSON
        import re
        # 提取JSON部分
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
            if 'sub_modifications' in result:
                return result['sub_modifications']
        
        # 如果JSON解析失败，尝试简单的文本解析
        lines = generated_text.split('\n')
        sub_mods = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('{') and not line.startswith('}') and not line.startswith('"sub_modifications"'):
                # 移除引号和逗号
                line = re.sub(r'^["\s]*', '', line)
                line = re.sub(r'[",\s]*$', '', line)
                if line:
                    sub_mods.append(line)
        
        return sub_mods if sub_mods else [generated_text]
    
    except Exception as e:
        print(f"Error parsing decomposition result: {e}")
        return [generated_text]


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
        [messages], 
        tokenize=False, 
        add_generation_prompt=True,
        add_vision_id=True,
    )
    
    # 处理图像信息
    image_inputs, video_inputs = process_vision_info([messages])
    
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
    """计算yes/no的logits并转换为概率分数"""
    
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
        
        # 清理中间变量
        del outputs, logits
        torch.cuda.empty_cache()
    
    return yes_probs


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
    parser = argparse.ArgumentParser(description='Decompose modification texts using Qwen2VL')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to Qwen2VL model directory')
    parser.add_argument('--json_file', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--output_file', type=str, default='decomposed_results.json', help='Output JSON file')
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
    processor = AutoProcessor.from_pretrained(args.model_dir)
    
    # 设置tokenizer的padding_side为'left'，以兼容Flash Attention
    processor.tokenizer.padding_side = 'left'
    
    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
    
    # 创建文本分解数据集和数据加载器
    decomp_dataset = TextDecompositionDataset(json_data)
    print('num of queries for decomposition: ', len(decomp_dataset))

    # 创建DataLoader用于文本分解
    decomp_dataloader = DataLoader(
        decomp_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 文本处理不需要多进程
        collate_fn=lambda batch: process_text_decomposition_batch(batch, processor)
    )
    
    # 使用accelerator准备dataloader
    decomp_dataloader = accelerator.prepare(decomp_dataloader)
    
    # 处理文本分解
    all_decomposition_results = []
    
    if accelerator.is_main_process:
        pbar = tqdm(total=len(decomp_dataloader), desc="Decomposing modification texts")
    
    for batch_inputs, batch_info in decomp_dataloader:
        try:
            # 生成分解结果
            generated_texts = generate_text_decomposition(model, batch_inputs, processor, device)
            
            # 解析结果
            for i, generated_text in enumerate(generated_texts):
                sub_modifications = parse_decomposition_result(generated_text)
                result = {
                    'query_id': batch_info[i]['query_id'],
                    'original_modification_text': batch_info[i]['original_modification_text'],
                    'sub_modifications': sub_modifications,
                    'generated_text': generated_text  # 保留原始生成文本用于调试
                }
                all_decomposition_results.append(result)
            
            if accelerator.is_main_process:
                pbar.update(1)
                
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM error in batch, skipping: {e}")
            torch.cuda.empty_cache()
            continue
    
    if accelerator.is_main_process:
        pbar.close()
    
    # 收集所有进程的结果
    all_decomposition_results = gather_object(all_decomposition_results)
    
    # 仅在主进程中保存结果
    if accelerator.is_main_process:
        print("Processing decomposition results...")
        
        # 创建query_id到分解结果的映射
        decomp_dict = {result['query_id']: result for result in all_decomposition_results}
        
        # 更新原始JSON数据，添加sub_modifications字段
        output_data = json_data.copy()
        
        for query in output_data['queries']:
            query_id = query['query_id']
            if query_id in decomp_dict:
                query['sub_modifications'] = decomp_dict[query_id]['sub_modifications']
                query['decomposition_generated_text'] = decomp_dict[query_id]['generated_text']
            else:
                query['sub_modifications'] = []
                query['decomposition_generated_text'] = ""
        
        # 添加元数据信息
        if 'metadata' not in output_data:
            output_data['metadata'] = {}
        
        output_data['metadata']['decomposed'] = True
        output_data['metadata']['decomposition_model'] = args.model_dir
        
        # 保存到文件
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDecomposition results saved to {args.output_file}")
        
        # 打印统计信息
        total_queries = len(output_data['queries'])
        queries_with_decomp = sum(1 for q in output_data['queries'] if q.get('sub_modifications'))
        avg_sub_mods = sum(len(q.get('sub_modifications', [])) for q in output_data['queries']) / total_queries if total_queries > 0 else 0
        
        print(f"Total queries: {total_queries}")
        print(f"Queries with decomposition: {queries_with_decomp}")
        print(f"Average sub-modifications per query: {avg_sub_mods:.2f}")


if __name__ == "__main__":
    main()

'''
运行命令示例:

# 基础运行命令
accelerate launch --num_processes 8 text_split.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2.5-VL-32B-Instruct \
    --json_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/cirr_retrieval_top10.json \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file R1_54_2_5_32b_decomposed_results.json \
    --batch_size 2

# 如果内存充足可以增大批处理大小
accelerate launch --num_processes 8 modified_rerank_script.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --json_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/retrieval_results/base_model_20250910_180548/cirr_retrieval_top10.json \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file decomposed_results.json \
    --batch_size 4
'''