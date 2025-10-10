# aug/batchers.py
from src.utils import print_rank


class CaptionBatcher:
    """
    封装不同 backbone 的批处理生成逻辑
    """

    def __init__(self, foundation_model, model_args, prepare_fns, generate_fns):
        """
        Args:
            foundation_model: 底层生成模型 (Qwen / LLaVA / Generic)
            model_args: 模型参数配置
            prepare_fns: dict, 包含 prepare_xxx_inputs 的函数
            generate_fns: dict, 包含 generate_with_xxx 的函数
        """
        self.foundation_model = foundation_model
        self.model_args = model_args
        self.prepare_fns = prepare_fns
        self.generate_fns = generate_fns

    def generate_batch(self, ref_images, target_images, original_texts, processor, device):
        """根据 backbone 类型批量生成 captions"""
        backbone = getattr(self.model_args, "foundation_model_backbone", "qwen2_vl")

        if backbone in ["qwen2_vl", "qwen", "qwen2_5_vl"]:
            return self._generate_qwen_batch(
                ref_images, target_images, original_texts, processor, device
            )
        elif backbone in ["llava", "llava_next"]:
            return self._generate_llava_batch(
                ref_images, target_images, original_texts, processor, device
            )
        else:
            return self._generate_generic_batch(
                ref_images, target_images, original_texts, processor, device
            )

    def _generate_qwen_batch(self, ref_images, target_images, prompts, processor, device):
        """批量使用 Qwen 生成文本"""
        results = []
        for ref_img, tgt_img, prompt in zip(ref_images, target_images, prompts):
            try:
                inputs = self.prepare_fns["qwen"](ref_img, tgt_img, prompt, processor, device)
                text = self.generate_fns["qwen"](inputs, device, self.foundation_model)
                results.append(text)
            except Exception as e:
                print_rank(f"Error in Qwen batch: {e}")
                results.append(None)
        return results

    def _generate_llava_batch(self, ref_images, target_images, prompts, processor, device):
        """批量使用 LLaVA 生成文本"""
        results = []
        for ref_img, tgt_img, prompt in zip(ref_images, target_images, prompts):
            try:
                inputs = self.prepare_fns["llava"](ref_img, tgt_img, prompt, processor, device)
                text = self.generate_fns["llava"](inputs, device, self.foundation_model)
                results.append(text)
            except Exception as e:
                print_rank(f"Error in LLaVA batch: {e}")
                results.append(None)
        return results

    def _generate_generic_batch(self, ref_images, target_images, prompts, processor, device):
        """批量使用通用模型生成文本"""
        results = []
        for ref_img, tgt_img, prompt in zip(ref_images, target_images, prompts):
            try:
                inputs = self.prepare_fns["generic"](ref_img, tgt_img, prompt, processor, device)
                text = self.generate_fns["generic"](inputs, device, self.foundation_model)
                results.append(text)
            except Exception as e:
                print_rank(f"Error in Generic batch: {e}")
                results.append(None)
        return results
