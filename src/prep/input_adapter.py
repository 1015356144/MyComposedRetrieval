# src/prep/input_adapter.py
import os
from typing import List, Dict, Any, Optional
from PIL import Image

from src.utils import print_rank
from src.model.processor import process_vlm_inputs_fns, process_input_text

class VLMInputPreparer:
    """
    兼容你旧签名的前处理适配器：
    - 与 dataset 解耦，只负责把(路径/文本) → 模型前向 inputs
    - 提供与旧版一致的三类函数：
        _prepare_vlm_inputs(...)
        prepare_target_inputs(...)
        prepare_query_inputs(...)
    """

    def __init__(
        self,
        image_base_dir: str = "",
        default_processor=None,
        default_backbone: Optional[str] = None,
        default_device: Optional[str] = None,
    ):
        self.image_base_dir = image_base_dir
        self.default_processor = default_processor
        self.default_backbone = default_backbone or "qwen2_vl"
        self.default_device = default_device

    # -------- path helper --------
    def _resolve(self, path: str) -> str:
        if not isinstance(path, str):
            return str(path)
        if path.startswith("./"):
            return os.path.join(self.image_base_dir, path[2:])
        if os.path.isabs(path):
            return path
        return os.path.join(self.image_base_dir, path)

    # -------- core (兼容旧签名) --------
    def _prepare_vlm_inputs(
        self,
        image_paths: List[str],
        texts: List[str],
        processor=None,
        model_backbone: Optional[str] = None,
        device: Optional[str] = None,
        input_type: str = "general",
    ) -> Dict[str, Any]:
        """
        等价于你旧版 _prepare_vlm_inputs(...)
        """
        proc = processor or self.default_processor
        backbone = model_backbone or self.default_backbone
        dev = device or self.default_device

        if proc is None:
            raise ValueError("processor is None. Provide `processor` or set default_processor in VLMInputPreparer.")

        images, processed_texts = [], []

        for img_path, text in zip(image_paths, texts):
            try:
                full_path = self._resolve(img_path)
                img = Image.open(full_path).convert("RGB")
            except Exception as e:
                print_rank(f"Error loading image {img_path}: {e}")
                img = Image.new("RGB", (224, 224), color="white")
            images.append(img)
            processed_texts.append(text if text is not None else "")

        try:
            model_inputs = {"text": processed_texts, "images": images}
            if backbone in process_vlm_inputs_fns:
                inputs = process_vlm_inputs_fns[backbone](model_inputs, proc)
            else:
                # 兜底：直接用 HF processor
                inputs = proc(text=processed_texts, images=images, return_tensors="pt")

            # 上设备
            if dev is not None:
                for k, v in list(inputs.items()):
                    try:
                        inputs[k] = v.to(dev)
                    except Exception:
                        pass
            return inputs

        except Exception as e:
            print_rank(f"Error in VLM2Vec processor for {input_type}: {e}")
            raise

    def prepare_target_inputs(
        self,
        target_paths: List[str],
        processor=None,
        model_backbone: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        等价于你旧版 _prepare_target_inputs(...)
        为目标图片构造含<image>占位的简短文本（与旧逻辑一致）
        """
        backbone = model_backbone or self.default_backbone
        texts = [
            process_input_text(
                instruction="Represent the given image",
                model_backbone=backbone,
                text="",
                add_image_token=True,
            )
            for _ in target_paths
        ]
        return self._prepare_vlm_inputs(
            target_paths, texts, processor=processor, model_backbone=backbone, device=device, input_type="target"
        )

    def prepare_query_inputs(
        self,
        batch: List[Dict[str, Any]],
        processor=None,
        model_backbone: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        等价于你旧版 _prepare_query_inputs(batch, ...)
        batch: [{'reference_image': ..., 'modification_text': ...}, ...]
        """
        backbone = model_backbone or self.default_backbone
        image_paths = [item["reference_image"] for item in batch]
        texts = [
            process_input_text(
                instruction="Represent the given image with the following modification",
                model_backbone=backbone,
                text=item["modification_text"],
                add_image_token=True,
            )
            for item in batch
        ]
        return self._prepare_vlm_inputs(
            image_paths, texts, processor=processor, model_backbone=backbone, device=device, input_type="query"
        )