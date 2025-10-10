#!/usr/bin/env python3
"""
CIRR检索脚本 - 加载模型对CIRR验证集进行检索并保存top-k结果
基于现有评估代码改编，专门用于检索和保存结果
"""

import os
import sys
import json
import torch
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, get_backbone_name, VLM_IMAGE_TOKENS
from src.utils import print_rank, print_master

logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class CIRRRetrievalArguments:
    """CIRR检索专用参数"""
    model_path: str = field(
        metadata={"help": "训练好的模型检查点路径 (可以是checkpoint-xxx或iteration_x目录)"}
    )
    base_model_name: str = field(
        default=None,
        metadata={"help": "基础模型名称 (例如 Qwen/Qwen2-VL-2B-Instruct). 如果不提供，将尝试从model_path推断"}
    )
    output_file: str = field(
        default=None,
        metadata={"help": "保存检索结果的JSON文件路径. 如果不提供，将自动生成基于模型路径的文件名"}
    )
    top_k: int = field(
        default=10,
        metadata={"help": "保存每个查询的top-k检索结果 (默认: 10)"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "批处理大小 (默认: 8)"}
    )
    device: str = field(
        default="auto",
        metadata={"help": "使用的设备: 'auto', 'cuda', 'cuda:0', 等"}
    )
    cirr_data_dir: str = field(
        default=None,
        metadata={"help": "CIRR数据集目录路径"}
    )
    cirr_image_dir: str = field(
        default=None,
        metadata={"help": "CIRR图像目录路径"}
    )
    save_embeddings: bool = field(
        default=False,
        metadata={"help": "是否同时保存查询和候选图像的嵌入向量"}
    )


class CIRRRetriever:
    """
    CIRR检索器 - 基于CIRREvaluator改编，专门用于检索和保存结果
    """
    
    def __init__(self, 
                 model,
                 processor, 
                 data_args,
                 model_args,
                 device='cuda',
                 batch_size=8,
                 cirr_data_dir=None,
                 cirr_image_dir=None):
        self.model = model
        self.processor = processor
        self.data_args = data_args
        self.model_args = model_args
        self.device = device
        self.batch_size = batch_size
        
        # Get model backbone
        self.model_backbone = getattr(model_args, 'model_backbone', 'qwen2_vl')
        
        # Configure CIRR data paths
        self._configure_data_paths(cirr_data_dir, cirr_image_dir)
        
        # Load CIRR test data
        self.test_data, self.candidate_images = self._load_cirr_test_data()
        
        print_master(f"加载了 {len(self.test_data)} 个查询")
        print_master(f"加载了 {len(self.candidate_images)} 个候选图像")
    
    def _configure_data_paths(self, cirr_data_dir=None, cirr_image_dir=None):
        """配置CIRR数据集路径"""
        # 使用提供的路径或默认路径
        if cirr_data_dir:
            self.data_dir = cirr_data_dir
        else:
            self.data_dir = '/home/guohaiyun/yty_data/CIRR/cirr'
        
        if cirr_image_dir:
            self.image_base_dir = cirr_image_dir
        else:
            self.image_base_dir = '/home/guohaiyun/yty_data/CIRR'
        
        # 设置文件路径
        self.captions_file = os.path.join(self.data_dir, 'captions/cap.rc2.val.json')
        self.image_splits_file = os.path.join(self.data_dir, 'image_splits/split.rc2.val.json')
        
        print_master(f"使用CIRR数据目录: {self.data_dir}")
        print_master(f"使用CIRR图像目录: {self.image_base_dir}")
    

    def _load_cirr_test_data(self) -> Tuple[List[Dict], List[str]]:
        """加载CIRR验证数据"""
        try:
            if not os.path.exists(self.captions_file):
                print_master(f"警告: CIRR验证查询文件未找到 {self.captions_file}")
                return self._create_dummy_test_data()
            
            # 加载验证查询
            with open(self.captions_file, 'r') as f:
                val_queries = json.load(f)
            
            # 加载验证图像分割信息
            if os.path.exists(self.image_splits_file):
                with open(self.image_splits_file, 'r') as f:
                    val_splits = json.load(f)
                candidate_images = list(val_splits.keys())
                self.image_splits = val_splits
                print_master(f"从验证分割中加载了 {len(candidate_images)} 个候选图像")
            else:
                print_master(f"警告: 验证分割文件未找到 {self.image_splits_file}")
                candidate_images = [f"dummy_img_{i}" for i in range(100)]
                self.image_splits = {}
            
            print_master(f"加载了 {len(val_queries)} 个CIRR验证查询")
            return val_queries, candidate_images
            
        except Exception as e:
            print_master(f"加载CIRR验证数据时出错: {e}")
            return self._create_dummy_test_data()
    
    def _create_dummy_test_data(self) -> Tuple[List[Dict], List[str]]:
        """创建虚拟测试数据"""
        dummy_data = []
        for i in range(50):
            dummy_data.append({
                'pairid': i,
                'reference': f'dummy_ref_{i}',
                'target_hard': f'dummy_target_{i}',
                'caption': f'虚拟修改文本 {i}',
                'target_soft': {},
                'img_set': {'members': [f'dummy_img_{j}' for j in range(i, i+5)]}
            })
        candidate_images = [f"dummy_img_{i}" for i in range(100)]
        self.image_splits = {}
        return dummy_data, candidate_images
    
    
    def _encode_batch(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """
        直接复用CIRREvaluator的编码逻辑，避免重复实现
        """
        try:
            # 创建临时CIRREvaluator实例来复用其编码逻辑
            from src.evaluation.cirr_evaluator import CIRREvaluator
            
            # 创建临时evaluator实例，复用其成熟的编码方法
            temp_evaluator = CIRREvaluator(
                model=self.model,
                processor=self.processor,
                data_args=self.data_args,
                model_args=self.model_args,
                device=self.device,
                batch_size=self.batch_size
            )
            
            # 直接调用evaluator的编码方法
            return temp_evaluator._encode_batch(batch_data)
                
        except Exception as e:
            print_master(f"编码批次时出错: {e}")
            import traceback
            print_master(f"Traceback: {traceback.format_exc()}")
            # 返回零嵌入作为备用
            return torch.zeros(len(batch_data['text']), 512, device=self.device)
    
    def _encode_images(self, image_names: List[str]) -> torch.Tensor:
        """编码候选图像 - 复用CIRREvaluator的逻辑"""
        try:
            # 创建临时CIRREvaluator实例
            from src.evaluation.cirr_evaluator import CIRREvaluator
            temp_evaluator = CIRREvaluator(
                model=self.model,
                processor=self.processor,
                data_args=self.data_args,
                model_args=self.model_args,
                device=self.device,
                batch_size=self.batch_size
            )
            
            # 直接调用evaluator的编码方法
            return temp_evaluator._encode_images_local(image_names)
        except Exception as e:
            print_master(f"编码候选图像时出错: {e}")
            return torch.empty(0, 512, device=self.device)
    
    def _encode_composed_queries(self, queries: List[Dict]) -> torch.Tensor:
        """编码复合查询 - 复用CIRREvaluator的逻辑"""
        try:
            # 创建临时CIRREvaluator实例
            from src.evaluation.cirr_evaluator import CIRREvaluator
            temp_evaluator = CIRREvaluator(
                model=self.model,
                processor=self.processor,
                data_args=self.data_args,
                model_args=self.model_args,
                device=self.device,
                batch_size=self.batch_size
            )
            
            # 直接调用evaluator的编码方法
            return temp_evaluator._encode_composed_queries_local(queries)
        except Exception as e:
            print_master(f"编码复合查询时出错: {e}")
            return torch.empty(0, 512, device=self.device)
    
    def retrieve_top_k(self, top_k: int = 10, save_embeddings: bool = False) -> Dict[str, Any]:
        """
        对所有查询进行检索并返回top-k结果
        
        Args:
            top_k: 每个查询返回的top-k结果数量
            save_embeddings: 是否保存嵌入向量
        
        Returns:
            包含所有检索结果的字典
        """
        print_master("开始CIRR检索...")
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 编码所有候选图像
        candidate_embeddings = self._encode_images(self.candidate_images)
        candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
        
        # 编码所有复合查询
        query_embeddings = self._encode_composed_queries(self.test_data)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        
        # 计算相似度
        print_master("计算相似度...")
        similarities = torch.mm(query_embeddings, candidate_embeddings.t())  # [num_queries, num_candidates]
        # 确保similarities为float32类型，避免BFloat16转numpy时出错
        similarities = similarities.float()
        
        # 对于每个查询，排除参考图像（避免自检索）
        print_master("排除参考图像...")
        for query_idx, query in enumerate(self.test_data):
            ref_image = query['reference']
            if ref_image in self.candidate_images:
                ref_idx = self.candidate_images.index(ref_image)
                similarities[query_idx, ref_idx] = -float('inf')
        
        # 获取top-k结果
        print_master(f"获取每个查询的top-{top_k}结果...")
        _, top_k_indices = torch.topk(similarities, k=top_k, dim=1, largest=True)
        
        # 构建结果
        results = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_path': getattr(self.model_args, 'checkpoint_path', 'unknown'),
                'model_backbone': self.model_backbone,
                'total_queries': len(self.test_data),
                'total_candidates': len(self.candidate_images),
                'top_k': top_k,
                'batch_size': self.batch_size,
                'device': str(self.device)
            },
            'queries': [],
            'candidate_images': self.candidate_images
        }
        
        # 如果需要保存嵌入向量
        if save_embeddings:
            results['embeddings'] = {
                'query_embeddings': query_embeddings.cpu().numpy().tolist(),
                'candidate_embeddings': candidate_embeddings.cpu().numpy().tolist()
            }
        
        # 为每个查询构建详细结果
        print_master("构建详细结果...")
        for query_idx, query in enumerate(tqdm(self.test_data, desc="处理查询结果")):
            # 获取top-k候选图像索引
            top_k_idx = top_k_indices[query_idx].cpu().numpy()
            
            # 获取对应的相似度分数
            top_k_scores = similarities[query_idx, top_k_idx].cpu().numpy()
            
            # 构建top-k结果列表
            retrieval_results = []
            for rank, (candidate_idx, score) in enumerate(zip(top_k_idx, top_k_scores)):
                candidate_name = self.candidate_images[candidate_idx]
                retrieval_results.append({
                    'rank': rank + 1,
                    'candidate_image': candidate_name,
                    'similarity_score': float(score),
                    'candidate_index': int(candidate_idx)
                })
            
            # 构建查询结果
            query_result = {
                'query_id': query_idx,
                'pairid': query['pairid'],
                'reference_image': query['reference'],
                'target_hard': query['target_hard'],
                'modification_text': query['caption'],
                'target_soft': query.get('target_soft', {}),
                'img_set': query.get('img_set', {}),
                'retrieval_results': retrieval_results
            }
            
            # 添加ground truth信息
            if 'target_hard' in query:
                # 检查target_hard是否在top-k结果中
                target_found = False
                target_rank = None
                for result in retrieval_results:
                    if result['candidate_image'] == query['target_hard']:
                        target_found = True
                        target_rank = result['rank']
                        break
                
                query_result['ground_truth'] = {
                    'target_hard': query['target_hard'],
                    'found_in_top_k': target_found,
                    'rank_in_top_k': target_rank
                }
            
            results['queries'].append(query_result)
        
        # 计算一些基本统计信息
        if 'ground_truth' in results['queries'][0]:
            found_count = sum(1 for q in results['queries'] if q['ground_truth']['found_in_top_k'])
            accuracy_at_k = found_count / len(results['queries'])
            results['metadata']['accuracy_at_k'] = accuracy_at_k
            results['metadata']['found_in_top_k_count'] = found_count
            print_master(f"Accuracy@{top_k}: {accuracy_at_k:.4f} ({found_count}/{len(results['queries'])})")
        
        print_master("检索完成!")
        return results


def setup_device(device_arg: str) -> str:
    """设置并返回适当的设备"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print_master(f"使用CUDA设备: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print_master("使用CPU设备")
    else:
        device = device_arg
        print_master(f"使用指定设备: {device}")
    
    return device


def infer_model_name_from_path(model_path: str) -> str:
    """从检查点路径推断基础模型名称（与 eval_cirr.py 一致）"""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            for key in ['_name_or_path', 'name_or_path', 'model_name', 'base_model_name']:
                if key in config and config[key]:
                    model_name = config[key]
                    print_master(f"从config推断基础模型名称: {model_name}")
                    return model_name
        except Exception as e:
            print_master(f"警告: 读取config.json失败: {e}")
    path_lower = model_path.lower()
    if "qwen2-vl" in path_lower:
        if "2b" in path_lower:
            return "Qwen/Qwen2-VL-2B-Instruct"
        elif "7b" in path_lower:
            return "Qwen/Qwen2-VL-7B-Instruct"
    elif "llava" in path_lower:
        if "7b" in path_lower:
            return "llava-hf/llava-1.5-7b-hf"
        elif "13b" in path_lower:
            return "llava-hf/llava-1.5-13b-hf"
    print_master("警告: 无法推断基础模型名称，使用默认值")
    return "Qwen/Qwen2-VL-2B-Instruct"


def load_model_and_processor(retrieval_args: CIRRRetrievalArguments, model_args: ModelArguments, data_args: DataArguments):
    """加载模型和处理器（复制 eval_cirr.py 的逻辑，保证一致性与稳定性）"""
    print_master("=" * 60)
    print_master("加载模型和处理器")
    print_master("=" * 60)

    # 1) 基础模型名确定
    if retrieval_args.base_model_name:
        model_name = retrieval_args.base_model_name
        print_master(f"使用提供的基础模型名称: {model_name}")
    else:
        model_name = infer_model_name_from_path(retrieval_args.model_path)
        print_master(f"推断的基础模型名称: {model_name}")

    # 2) 处理 auto-infer
    if model_args.model_name == "auto-infer":
        model_args.model_name = model_name
        print_master(f"自动推断的 model_name: {model_name}")
    elif not model_args.model_name:
        model_args.model_name = model_name

    model_args.checkpoint_path = retrieval_args.model_path

    # 3) 覆盖关键默认值以对齐训练
    print_master("覆盖ModelArguments默认值以匹配训练配置...")
    model_args.pooling = 'eos'
    model_args.normalize = True
    print_master(f"✅ 设置 pooling={model_args.pooling}, normalize={model_args.normalize}")
    print_master(f"✅ 设置 lora_r={model_args.lora_r}, lora_dropout={model_args.lora_dropout}")

    data_args.max_len = 512
    data_args.resize_max_pixels = 262144 # 512*512
    print_master(f"✅ 设置 max_len={data_args.max_len}, resize_max_pixels={data_args.resize_max_pixels}")

    # 4) LoRA / 本地配置检测，与 eval_cirr.py 一致
    local_config_path = os.path.join(retrieval_args.model_path, "config.json")
    adapter_config_path = os.path.join(retrieval_args.model_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        print_master(f"找到LoRA适配器配置: {adapter_config_path}")
        try:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            if 'base_model_name_or_path' in adapter_config:
                base_model_name = adapter_config['base_model_name_or_path']
                print_master(f"LoRA基础模型: {base_model_name}")
                model_args.model_name = base_model_name
                model_args.lora = True
                model_args.checkpoint_path = retrieval_args.model_path
                # 依据基础模型名做简单的backbone推断（与 eval_cirr.py 保持一致）
                lower = base_model_name.lower()
                if 'qwen2' in lower:
                    setattr(model_args, 'model_backbone', 'qwen2_vl')
                elif 'llava' in lower:
                    setattr(model_args, 'model_backbone', 'llava_next')
                else:
                    setattr(model_args, 'model_backbone', 'qwen2_vl')
                print_master(f"LoRA backbone: {model_args.model_backbone}")
            else:
                print_master("警告: adapter_config 中未找到 base_model_name_or_path，使用默认backbone qwen2_vl")
                setattr(model_args, 'model_backbone', 'qwen2_vl')
        except Exception as e:
            print_master(f"读取adapter_config失败，使用默认backbone qwen2_vl: {e}")
            setattr(model_args, 'model_backbone', 'qwen2_vl')
    elif os.path.exists(local_config_path):
        print_master(f"找到本地config.json: {local_config_path}")
        try:
            with open(local_config_path, 'r') as f:
                local_config = json.load(f)
            if 'model_type' in local_config:
                mt = local_config['model_type'].lower()
                if 'qwen2_vl' in mt:
                    setattr(model_args, 'model_backbone', 'qwen2_vl')
                elif 'llava' in mt:
                    setattr(model_args, 'model_backbone', 'llava_next')
                else:
                    setattr(model_args, 'model_backbone', 'qwen2_vl')
                print_master(f"从本地config推断backbone: {model_args.model_backbone}")
            else:
                setattr(model_args, 'model_backbone', 'qwen2_vl')
                print_master(f"使用默认backbone: {model_args.model_backbone}")
        except Exception as e:
            setattr(model_args, 'model_backbone', 'qwen2_vl')
            print_master(f"读取本地config失败，默认backbone: {e}")
    else:
        print_master("未找到本地config或adapter_config，使用默认backbone qwen2_vl")
        setattr(model_args, 'model_backbone', 'qwen2_vl')

    # 5) 按 eval 脚本方式加载模型
    model = None
    if getattr(model_args, 'lora', False):
        print_master("加载LoRA模型:")
        print_master(f"  基础模型: {model_args.model_name}")
        print_master(f"  LoRA检查点: {model_args.checkpoint_path}")
        try:
            model = MMEBModel.load(model_args, is_trainable=False)
            model.eval()
            print_master("✅ LoRA模型加载成功")
        except Exception as e:
            print_master(f"❌ 加载LoRA模型失败: {e}")
            raise
    else:
        print_master(f"从本地检查点加载完整模型: {retrieval_args.model_path}")
        model_args.checkpoint_path = retrieval_args.model_path
        try:
            print_master("使用 MMEBModel.load 加载（与训练一致）...")
            model = MMEBModel.load(model_args, is_trainable=False)
            model.eval()
            print_master("✅ 完整模型加载成功")
        except Exception as e:
            print_master(f"❌ 完整模型加载失败: {e}")
            raise

    # 6) 加载处理器
    print_master("加载处理器...")
    try:
        processor = load_processor(model_args, data_args)
        print_master("✅ 处理器加载成功")
    except Exception as e:
        print_master(f"❌ 加载处理器失败: {e}")
        raise

    setattr(model, 'processor', processor)
    print_master("=" * 60)
    return model, processor


def generate_output_filename(retrieval_args: CIRRRetrievalArguments) -> str:
    """生成输出文件名（默认保存在项目根目录下的 ./retrieval_results/ 子目录中）"""
    if retrieval_args.output_file:
        return retrieval_args.output_file
    
    # 基于脚本所在目录，生成相对项目根目录的输出路径
    project_root = os.path.dirname(__file__)
    base_dir = os.path.join(project_root, 'retrieval_results')
    
    # 从模型路径生成目录名
    model_path = retrieval_args.model_path
    model_name = os.path.basename(model_path.rstrip('/'))
    
    # 添加时间戳，自动创建新的子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{model_name}_{timestamp}")
    
    # 最终文件名
    filename = f"cirr_retrieval_top{retrieval_args.top_k}.json"
    
    return os.path.join(run_dir, filename)


def save_retrieval_results(results: Dict[str, Any], output_file: str):
    """保存检索结果到JSON文件"""
    try:
        # 创建输出目录
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print_master(f"✅ 检索结果已保存到: {output_file}")
        
        # 打印文件大小
        file_size = os.path.getsize(output_file)
        print_master(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print_master(f"❌ 保存检索结果失败: {e}")
        raise


def main():
    """主函数"""
    # 解析参数
    parser = HfArgumentParser((CIRRRetrievalArguments, ModelArguments, DataArguments))
    
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        # 从JSON配置文件加载
        retrieval_args, model_args, data_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        # 从命令行解析
        retrieval_args, model_args, data_args = parser.parse_args_into_dataclasses()
    
    # 验证参数
    if not retrieval_args.model_path:
        raise ValueError("model_path是必需的")
    
    if not os.path.exists(retrieval_args.model_path):
        raise ValueError(f"模型路径不存在: {retrieval_args.model_path}")
    
    # 设置设备
    device = setup_device(retrieval_args.device)
    
    # 加载模型和处理器
    model, processor = load_model_and_processor(retrieval_args, model_args, data_args)
    
    # 移动模型到设备
    model = model.to(device)
    print_master(f"模型已移动到设备: {device}")
    
    # 创建检索器
    retriever = CIRRRetriever(
        model=model,
        processor=processor,
        data_args=data_args,
        model_args=model_args,
        device=device,
        batch_size=retrieval_args.batch_size,
        cirr_data_dir=retrieval_args.cirr_data_dir,
        cirr_image_dir=retrieval_args.cirr_image_dir
    )
    
    # 执行检索
    results = retriever.retrieve_top_k(
        top_k=retrieval_args.top_k,
        save_embeddings=retrieval_args.save_embeddings
    )
    
    # 生成输出文件名
    output_file = generate_output_filename(retrieval_args)
    
    # 保存结果
    save_retrieval_results(results, output_file)
    
    print_master("🎉 CIRR检索完成!")
    
    return results


if __name__ == "__main__":
    main()

# 示例命令（仅供参考，请在终端执行，不要写在Python文件里）
# python retrieval_cirr.py \
#     --model_path /home/guohaiyun/yangtianyu/MyComposedRetrieval/experiments/IterativeCIRR_qwen2vl_20250918_191807/training_iter_0/checkpoint-1500 \
#     --base_model_name "Qwen/Qwen2-VL-7B-Instruct" \
#     --top_k 10 \
#     --batch_size 8 \
#     --device cuda \
#     --model_name auto-infer