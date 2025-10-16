#!/usr/bin/env python3
"""
CIRR 检索脚本
- 与新版 eval_cirr.py、cirr_evaluator.py 对齐
- 统一模型/处理器加载、设备设置、LoRA 检测、backbone 推断
- 通过 evaluator 的统一编码接口生成嵌入，计算相似度并保存 top-k 检索结果
"""

import os
import sys
import re
import json
import torch
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import torch.nn.functional as F

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, get_backbone_name
from src.utils import print_rank, print_master


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class CIRRRetrievalArguments:
    """CIRR 检索专用参数"""
    model_path: str = field(
        metadata={"help": "训练好的模型检查点路径 (可以是 checkpoint-xxx 或 iteration_x 目录)"}
    )
    base_model_name: str = field(
        default=None,
        metadata={"help": "基础模型名称 (例如 Qwen/Qwen2-VL-2B-Instruct). 若不提供，将从 model_path 推断"}
    )
    output_file: str = field(
        default=None,
        metadata={"help": "保存检索结果的 JSON 文件路径. 若不提供，将基于模型路径自动生成"}
    )
    top_k: int = field(
        default=10,
        metadata={"help": "保存每个查询的 top-k 检索结果 (默认: 10)"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "批处理大小 (默认: 8，可视显存调整)"}
    )
    device: str = field(
        default="auto",
        metadata={"help": "设备: 'auto', 'cuda', 'cuda:0', 等"}
    )
    cirr_data_dir: str = field(
        default=None,
        metadata={"help": "CIRR 数据集目录路径 (包含 captions/ 与 image_splits/)"}
    )
    cirr_image_dir: str = field(
        default=None,
        metadata={"help": "CIRR 图像根目录路径 (包含 dev/test/train 子目录)"}
    )
    save_embeddings: bool = field(
        default=False,
        metadata={"help": "是否同时保存查询与候选图像的嵌入向量 (JSON 体积较大时慎用)"}
    )
    distributed: bool = field(
        default=False,
        metadata={"help": "是否使用分布式 (torchrun) 进行多卡编码与检索"}
    )


class CIRRRetriever:
    """
    CIRR 检索器
    - 负责 CIRR 数据加载
    - 通过 CIRREvaluator 的编码管线计算嵌入
    - 计算相似度并导出 top-k 检索结果
    """

    def __init__(
        self,
        model,
        processor,
        data_args,
        model_args,
        device='cuda',
        batch_size=8,
        cirr_data_dir=None,
        cirr_image_dir=None,
        distributed: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.data_args = data_args
        self.model_args = model_args
        self.device = device
        self.batch_size = batch_size
        self.distributed = distributed

        # backbone 与评测保持一致
        self.model_backbone = getattr(model_args, 'model_backbone', 'qwen2_vl')

        # 配置 CIRR 数据路径并加载数据
        self._configure_data_paths(cirr_data_dir, cirr_image_dir)
        self.test_data, self.candidate_images = self._load_cirr_test_data()

        # 仅用于复用 evaluator 的统一编码接口
        try:
            from src.evaluation.cirr_evaluator import CIRREvaluator
            self._evaluator = CIRREvaluator(
                model=self.model,
                processor=self.processor,
                data_args=self.data_args,
                model_args=self.model_args,
                device=self.device,
                batch_size=self.batch_size,
            )
        except Exception as e:
            print_master(f"警告: 创建 CIRREvaluator 失败，将回退到简化编码: {e}")
            self._evaluator = None

        print_master(f"加载了 {len(self.test_data)} 个查询")
        print_master(f"加载了 {len(self.candidate_images)} 个候选图像")

    def _configure_data_paths(self, cirr_data_dir=None, cirr_image_dir=None):
        """配置 CIRR 数据集路径"""
        # 与 evaluator 默认保持一致
        self.data_dir = cirr_data_dir or '/home/guohaiyun/yty_data/CIRR/cirr'
        self.image_base_dir = cirr_image_dir or '/home/guohaiyun/yty_data/CIRR'

        # 文件路径
        self.captions_file = os.path.join(self.data_dir, 'captions/cap.rc2.val.json')
        self.image_splits_file = os.path.join(self.data_dir, 'image_splits/split.rc2.val.json')

        print_master(f"使用 CIRR 数据目录: {self.data_dir}")
        print_master(f"使用 CIRR 图像目录: {self.image_base_dir}")

    def _load_cirr_test_data(self) -> Tuple[List[Dict], List[str]]:
        """加载 CIRR 验证数据"""
        try:
            if not os.path.exists(self.captions_file):
                print_master(f"警告: 未找到验证查询文件 {self.captions_file}")
                return self._create_dummy_test_data()

            with open(self.captions_file, 'r') as f:
                val_queries = json.load(f)

            if os.path.exists(self.image_splits_file):
                with open(self.image_splits_file, 'r') as f:
                    val_splits = json.load(f)
                candidate_images = list(val_splits.keys())
                self.image_splits = val_splits
                print_master(f"从验证分割加载 {len(candidate_images)} 个候选图像")
            else:
                print_master(f"警告: 未找到验证分割文件 {self.image_splits_file}")
                candidate_images = [f"dummy_img_{i}" for i in range(100)]
                self.image_splits = {}

            print_master(f"加载 {len(val_queries)} 个验证查询")
            return val_queries, candidate_images
        except Exception as e:
            print_master(f"加载 CIRR 验证数据失败: {e}")
            return self._create_dummy_test_data()

    def _create_dummy_test_data(self) -> Tuple[List[Dict], List[str]]:
        """创建虚拟数据以便在无数据时调试"""
        dummy_data = []
        for i in range(50):
            dummy_data.append({
                'pairid': i,
                'reference': f'dummy_ref_{i}',
                'target_hard': f'dummy_target_{i}',
                'caption': f'虚拟修改文本 {i}',
                'target_soft': {},
                'img_set': {'members': [f'dummy_img_{j}' for j in range(i, i + 5)]},
            })
        candidate_images = [f"dummy_img_{i}" for i in range(100)]
        self.image_splits = {}
        return dummy_data, candidate_images

    def _encode_images(self, image_names: List[str]) -> torch.Tensor:
        """编码候选图像，优先复用 evaluator 统一编码流程"""
        if self._evaluator is not None and hasattr(self._evaluator, '_encode_images_local'):
            return self._evaluator._encode_images_local(image_names)
        # 回退：返回空张量避免崩溃
        print_master("警告: 回退到空图像嵌入，请检查 evaluator 实现是否可用")
        return torch.empty(0, 512, device=self.device)

    def _encode_composed_queries(self, queries: List[Dict]) -> torch.Tensor:
        """编码复合查询，优先复用 evaluator 统一编码流程"""
        if self._evaluator is not None and hasattr(self._evaluator, '_encode_composed_queries_local'):
            return self._evaluator._encode_composed_queries_local(queries)
        # 回退：返回空张量避免崩溃
        print_master("警告: 回退到空查询嵌入，请检查 evaluator 实现是否可用")
        return torch.empty(0, 512, device=self.device)

    def retrieve_top_k(self, top_k: int = 10, save_embeddings: bool = False) -> Dict[str, Any]:
        """
        对所有查询进行检索并返回 top-k 结果
        """
        print_master("开始 CIRR 检索...")
        self.model.eval()

        # 1) 编码候选图像与查询（优先使用分布式编码）
        use_dist = False
        if self.distributed and self._evaluator is not None:
            try:
                import torch.distributed as dist
                use_dist = dist.is_initialized() and dist.get_world_size() > 1 \
                           and hasattr(self._evaluator, '_encode_images_distributed') \
                           and hasattr(self._evaluator, '_encode_queries_distributed')
            except Exception:
                use_dist = False

        if use_dist:
            print_master("使用分布式编码候选图像...")
            candidate_embeddings = self._evaluator._encode_images_distributed()
            print_master("使用分布式编码查询...")
            query_embeddings = self._evaluator._encode_queries_distributed()
        else:
            candidate_embeddings = self._encode_images(self.candidate_images)
            if candidate_embeddings.numel() == 0:
                print_master("❌ 候选图像嵌入为空，检索中止")
                return {}
            query_embeddings = self._encode_composed_queries(self.test_data)
            if query_embeddings.numel() == 0:
                print_master("❌ 查询嵌入为空，检索中止")
                return {}

        # 归一化
        candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        # 2) 相似度
        print_master("计算相似度...")
        similarities = torch.mm(query_embeddings, candidate_embeddings.t()).float()

        # 3) 排除参考图像
        print_master("排除参考图像...")
        image_to_idx = {img: idx for idx, img in enumerate(self.candidate_images)}
        for q_idx, q in enumerate(self.test_data):
            ref = q.get('reference')
            if ref in image_to_idx:
                similarities[q_idx, image_to_idx[ref]] = -float('inf')

        # 4) top-k
        k = min(top_k, similarities.size(1))
        print_master(f"获取每个查询的 top-{k} 结果...")
        _, topk_idx = torch.topk(similarities, k=k, dim=1, largest=True)

        # 5) 组织结果
        results: Dict[str, Any] = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_path': getattr(self.model_args, 'checkpoint_path', 'unknown'),
                'model_backbone': getattr(self.model_args, 'model_backbone', 'unknown'),
                'total_queries': len(self.test_data),
                'total_candidates': len(self.candidate_images),
                'top_k': k,
                'batch_size': self.batch_size,
                'device': str(self.device),
                'distributed': bool(use_dist),
            },
            'queries': [],
            'candidate_images': self.candidate_images,
        }

        if save_embeddings:
            try:
                results['embeddings'] = {
                    'query_embeddings': query_embeddings.cpu().numpy().tolist(),
                    'candidate_embeddings': candidate_embeddings.cpu().numpy().tolist(),
                }
            except Exception as e:
                print_master(f"警告: 保存嵌入失败，已跳过 (原因: {e})")

        print_master("构建详细结果...")
        for q_idx, q in enumerate(self.test_data):
            indices = topk_idx[q_idx].tolist()
            scores = similarities[q_idx, indices].tolist()

            retrieval_results = []
            for rank, (ci, s) in enumerate(zip(indices, scores), start=1):
                retrieval_results.append({
                    'rank': rank,
                    'candidate_image': self.candidate_images[ci],
                    'similarity_score': float(s),
                    'candidate_index': int(ci),
                })

            qr: Dict[str, Any] = {
                'query_id': q_idx,
                'pairid': q.get('pairid', q_idx),
                'reference_image': q.get('reference'),
                'target_hard': q.get('target_hard'),
                'modification_text': q.get('caption'),
                'target_soft': q.get('target_soft', {}),
                'img_set': q.get('img_set', {}),
                'retrieval_results': retrieval_results,
            }

            tgt = q.get('target_hard')
            if tgt is not None:
                found = None
                for item in retrieval_results:
                    if item['candidate_image'] == tgt:
                        found = item['rank']
                        break
                qr['ground_truth'] = {
                    'target_hard': tgt,
                    'found_in_top_k': found is not None,
                    'rank_in_top_k': found,
                }

            results['queries'].append(qr)

        # 6) 简单统计
        if results['queries'] and 'ground_truth' in results['queries'][0]:
            found_cnt = sum(1 for q in results['queries'] if q['ground_truth']['found_in_top_k'])
            acc_at_k = found_cnt / len(results['queries']) if results['queries'] else 0.0
            results['metadata']['accuracy_at_k'] = acc_at_k
            results['metadata']['found_in_top_k_count'] = found_cnt
            print_master(f"Accuracy@{k}: {acc_at_k:.4f} ({found_cnt}/{len(results['queries'])})")

        print_master("检索完成!")
        return results


def setup_device(device_arg: str, distributed: bool = False) -> str:
    """设置并返回设备 (支持分布式 torchrun)"""
    # 分布式初始化（仅当设置了环境变量时）
    if distributed and ('RANK' in os.environ and 'WORLD_SIZE' in os.environ):
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = f"cuda:{local_rank}"
                print_master(f"分布式已初始化: rank {dist.get_rank()}/{dist.get_world_size()}")
                print_master(f"使用设备: {device}")
                return device
            else:
                print_master("CUDA 不可用，回退 CPU 模式")
        except Exception as e:
            print_master(f"分布式初始化失败: {e}，回退单卡模式")

    # 单机/单卡
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            try:
                print_master(f"使用 CUDA 设备: {torch.cuda.get_device_name()}")
            except Exception:
                print_master("使用 CUDA 设备")
        else:
            device = 'cpu'
            print_master("CUDA 不可用，使用 CPU")
    else:
        device = device_arg
        print_master(f"使用指定设备: {device}")
    return device


def infer_model_name_from_path(model_path: str, quiet: bool = False) -> str:
    """
    从检查点路径推断基础模型名称
    - 增强：支持 qwen2(.5|_5)?[-_]?vl 等多种写法
    - 若存在 config.json 优先读取
    """
    path_lower = model_path.lower()

    # 直接匹配 qwen2(.5|_5)?[-_]?vl
    if re.search(r"qwen2(\.5|_5)?[-_]?vl", path_lower):
        is_qwen25 = bool(re.search(r"qwen2(\.5|_5)", path_lower))
        size = None
        if '2b' in path_lower:
            size = '2B'
        elif '7b' in path_lower:
            size = '7B'
        elif '32b' in path_lower:
            size = '32B'
        base = f"Qwen2.5-VL-{size or '7B'}-Instruct" if is_qwen25 else f"Qwen2-VL-{size or '7B'}-Instruct"
        model_name = f"/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/{base}"
        if not quiet:
            print_master(f"从路径模式推断基础模型: {model_name}")
        return model_name

    # 读取本地 config.json
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            for key in ['_name_or_path', 'name_or_path', 'model_name', 'base_model_name']:
                if key in cfg and cfg[key]:
                    if not quiet:
                        print_master(f"从 config 推断基础模型: {cfg[key]}")
                    return cfg[key]
        except Exception as e:
            if not quiet:
                print_master(f"警告: 读取 config.json 失败: {e}")

    default_name = 'Qwen/Qwen2-VL-2B-Instruct'
    if not quiet:
        print_master("警告: 无法推断基础模型，使用默认值")
    return default_name


def load_model_and_processor(
    retrieval_args: CIRRRetrievalArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
):
    """加载模型与处理器，逻辑与 eval_cirr.py 对齐"""
    print_master("=" * 60)
    print_master("加载模型与处理器")
    print_master("=" * 60)

    # 优先从 LoRA 适配器读取基础模型
    base_model_name = None
    lora_mode = False

    adapter_config_path = os.path.join(retrieval_args.model_path, 'adapter_config.json')
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, 'r') as f:
                ad_cfg = json.load(f)
            if 'base_model_name_or_path' in ad_cfg:
                base_model_name = ad_cfg['base_model_name_or_path']
                lora_mode = True
                print_master(f"检测到 LoRA 适配器，基础模型: {base_model_name}")
        except Exception as e:
            print_master(f"读取 adapter_config.json 失败 (将回退推断): {e}")

    # 用户显式指定覆盖
    if retrieval_args.base_model_name:
        base_model_name = retrieval_args.base_model_name
        print_master(f"使用提供的基础模型名称: {base_model_name}")

    # 推断
    if base_model_name is None:
        base_model_name = infer_model_name_from_path(retrieval_args.model_path, quiet=True)
        print_master(f"推断的基础模型名称: {base_model_name}")

    # 设置 model_args.model_name
    if model_args.model_name in [None, 'auto-infer']:
        model_args.model_name = base_model_name
        print_master(f"最终 model_name: {model_args.model_name}")

    # checkpoint 路径与 LoRA 标记
    model_args.checkpoint_path = retrieval_args.model_path
    model_args.lora = lora_mode or getattr(model_args, 'lora', False)

    # 关键默认值与训练对齐
    print_master("覆盖 ModelArguments 默认值以匹配训练配置...")
    model_args.pooling = 'eos'
    model_args.normalize = True
    print_master(f"✅ 设置 pooling={model_args.pooling}, normalize={model_args.normalize}")

    data_args.max_len = 512
    # 与新版评测默认分辨率一致 (384x384 = 147456)
    data_args.resize_max_pixels = 147456
    print_master(f"✅ 设置 max_len={data_args.max_len}, resize_max_pixels={data_args.resize_max_pixels}")

    # backbone 检测：优先 AutoConfig + get_backbone_name
    try:
        from transformers import AutoConfig
        base_cfg = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        detected_backbone = get_backbone_name(base_cfg, getattr(model_args, 'model_type', None))
        model_args.model_backbone = detected_backbone
        print_master(f"检测到 backbone: {detected_backbone}")
    except Exception as e_det:
        bl = model_args.model_name.lower()
        if 'qwen2.5' in bl or 'qwen2_5' in bl:
            model_args.model_backbone = 'qwen2_5_vl'
        elif 'qwen2' in bl:
            model_args.model_backbone = 'qwen2_vl'
        elif 'llava' in bl:
            model_args.model_backbone = 'llava_next'
        else:
            model_args.model_backbone = 'qwen2_vl'
        print_master(f"backbone 检测回退 ({e_det}): {model_args.model_backbone}")

    # 构建 / 加载模型
    model = None
    if model_args.lora:
        print_master("加载 LoRA 模型 (base + adapter)...")
        try:
            model = MMEBModel.load(model_args, is_trainable=False)
            model.eval()
            print_master("✅ LoRA 模型加载成功")
        except Exception as e:
            print_master(f"❌ LoRA 模型加载失败: {e}")
            raise
    else:
        print_master("从本地检查点加载完整模型 (非 LoRA)...")
        try:
            model = MMEBModel.load(model_args, is_trainable=False)
            model.eval()
            print_master("✅ 完整模型加载成功")
        except Exception as e:
            print_master(f"MMEBModel.load 失败: {e}")
            print_master("尝试 build + 手动加载权重 回退方案...")
            try:
                original_ckpt = model_args.checkpoint_path
                model_args.checkpoint_path = None
                model = MMEBModel.build(model_args)

                # 查找常见权重文件
                weight_file = None
                if os.path.isdir(retrieval_args.model_path):
                    for f in ["pytorch_model.bin", "model.safetensors", "model.bin"]:
                        fp = os.path.join(retrieval_args.model_path, f)
                        if os.path.exists(fp):
                            weight_file = fp
                            break
                if weight_file is None:
                    raise ValueError(f"未在 {retrieval_args.model_path} 找到权重文件")

                print_master(f"从 {weight_file} 加载权重...")
                if weight_file.endswith('.safetensors'):
                    from safetensors import safe_open
                    sd = {}
                    with safe_open(weight_file, framework='pt', device='cpu') as sf:
                        for k in sf.keys():
                            sd[k] = sf.get_tensor(k)
                else:
                    sd = torch.load(weight_file, map_location='cpu')

                model.load_state_dict(sd, strict=False)
                print_master("✅ 权重已加载到构建的模型中")
                model_args.checkpoint_path = original_ckpt
            except Exception as e2:
                print_master(f"❌ 回退方案失败: {e2}")
                raise

    # 处理器
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
    """生成输出文件路径 (默认写入 ./retrieval_results/<model>_<ts>/cirr_retrieval_topK.json)"""
    if retrieval_args.output_file:
        return retrieval_args.output_file

    project_root = os.path.dirname(__file__)
    base_dir = os.path.join(project_root, 'retrieval_results')

    model_path = retrieval_args.model_path.rstrip('/')
    model_name = os.path.basename(model_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{model_name}_{timestamp}")
    filename = f"cirr_retrieval_top{retrieval_args.top_k}.json"
    return os.path.join(run_dir, filename)


def save_retrieval_results(results: Dict[str, Any], output_file: str):
    """保存检索结果到 JSON 文件"""
    try:
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print_master(f"检索结果已保存: {output_file}")
    except Exception as e:
        print_master(f"❌ 保存检索结果失败: {e}")
        print_master(f"Traceback: {traceback.format_exc()}")


def main():
    """主流程：解析参数 -> 设置设备 -> 加载模型/处理器 -> 检索 -> 保存结果"""
    parser = HfArgumentParser((CIRRRetrievalArguments, ModelArguments, DataArguments))

    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        retrieval_args, model_args, data_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        retrieval_args, model_args, data_args = parser.parse_args_into_dataclasses()

    if not retrieval_args.model_path or not os.path.exists(retrieval_args.model_path):
        raise ValueError(f"无效的模型路径: {retrieval_args.model_path}")

    device = setup_device(retrieval_args.device, retrieval_args.distributed)

    model, processor = load_model_and_processor(retrieval_args, model_args, data_args)
    model = model.to(device)
    print_master(f"模型已移动到设备: {device}")

    retriever = CIRRRetriever(
        model=model,
        processor=processor,
        data_args=data_args,
        model_args=model_args,
        device=device,
        batch_size=retrieval_args.batch_size,
        cirr_data_dir=retrieval_args.cirr_data_dir,
        cirr_image_dir=retrieval_args.cirr_image_dir,
        distributed=retrieval_args.distributed,
    )

    results = retriever.retrieve_top_k(top_k=retrieval_args.top_k, save_embeddings=retrieval_args.save_embeddings)

    # 仅主进程保存
    is_main = True
    if retrieval_args.distributed:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                is_main = (dist.get_rank() == 0)
        except Exception:
            is_main = True

    if results and is_main:
        output_file = generate_output_filename(retrieval_args)
        save_retrieval_results(results, output_file)
        print_master("🎉 CIRR 检索完成！")

    # 分布式清理
    if retrieval_args.distributed:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except Exception:
            pass

    return results if results else {}


if __name__ == "__main__":
    main()

# 运行示例（在 bash 终端执行，以下命令已注释）
# 1) 单卡 - 完整权重目录（非 LoRA）
# python retrieval_cirr.py \
#   --model_path /home/guohaiyun/yangtianyu/MyComposedRetrieval/experiments/IterativeCIRR_qwen2vl_20250918_191807/training_iter_0/checkpoint-1500 \
#   --device cuda \
#   --top_k 10 \
#   --batch_size 8 \
#   --model_name auto-infer
#
# 2) 单卡 - LoRA 适配器目录（adapter_config.json 含 base_model_name_or_path）
# python retrieval_cirr.py \
#   --model_path /path/to/lora_adapter \
#   --device cuda \
#   --model_name auto-infer
#
#    如果 adapter_config.json 缺少 base_model_name_or_path，手动指定基础模型：
# python retrieval_cirr.py \
#   --model_path /path/to/lora_adapter \
#   --base_model_name Qwen/Qwen2-VL-7B-Instruct \
#   --device cuda \
#   --model_name auto-infer
#
# 3) 自定义数据路径与保存嵌入（文件会较大）
# python retrieval_cirr.py \
#   --model_path /path/to/checkpoint-1500 \
#   --cirr_data_dir /home/guohaiyun/yty_data/CIRR/cirr \
#   --cirr_image_dir /home/guohaiyun/yty_data/CIRR \
#   --save_embeddings \
#   --device cuda \
#   --model_name auto-infer
#
# 4) 多卡分布式（例如 8 卡）
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 retrieval_cirr.py \
#   --model_path /path/to/checkpoint-1500 \
#   --device cuda \
#   --batch_size 8 \
#   --top_k 50 \
#   --distributed True \
#   --model_name auto-infer