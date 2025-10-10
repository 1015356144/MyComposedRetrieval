# aug/validators.py
from src.utils import print_rank


class CaptionValidator:
    """
    Caption 有效性检查和过滤逻辑
    """

    def __init__(self):
        self.invalid_patterns = [
            "uncertain",
            "Generate Captions",
            "Generate Text_new",
            "Generate Polar Questions",
            "Answer Questions via Image Comparison",
            "Local Edits",
            "Decompose Intents",
        ]

    def is_valid(self, caption_text: str) -> bool:
        """检查生成的 caption 是否有效"""
        if not caption_text or not caption_text.strip():
            return False
        if caption_text == " " or len(caption_text) > 300:
            return False
        caption_lower = caption_text.lower()
        for pattern in self.invalid_patterns:
            if caption_lower == pattern or caption_lower.startswith(pattern):
                return False
        return True

    def filter_valid_samples(self, augmented_samples):
        """过滤掉无效 caption"""
        if not augmented_samples:
            return []

        valid_samples = []
        for sample in augmented_samples:
            if self.is_valid(sample.get("modification_text", "")):
                valid_samples.append(sample)
            else:
                # 调试用，仅打印前几个
                if len(valid_samples) < 5:
                    print_rank(
                        f"Filtered invalid caption: '{sample.get('modification_text', '')}' "
                        f"(original: '{sample.get('original_mod_text', '')}')"
                    )

        filtered_count = len(augmented_samples) - len(valid_samples)
        if filtered_count > 0:
            print_rank(
                f"Filtered out {filtered_count} invalid captions "
                f"({filtered_count/len(augmented_samples)*100:.1f}%)"
            )
        print_rank(
            f"Remaining valid samples: {len(valid_samples)}/{len(augmented_samples)}"
        )
        return valid_samples
