import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import argparse
import os

def _pick_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def load_model(model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
    """加载 Qwen2.5-VL-7B 模型与处理器（单图/多图/视频均可）"""
    print("正在加载模型...")
    dtype = _pick_dtype()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        # 环境支持可开启 FlashAttention-2 以省显存、提速
        # attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    print("模型加载完成!")
    return model, processor

def process_image_and_text(model, processor, image_path, question):
    """处理单张图片 + 文本，生成回答"""

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        print(f"图片加载成功: {image_path}  尺寸: {image.size}")
    except Exception as e:
        raise Exception(f"无法加载图片: {e}")

    # 构建对话消息（Qwen2.5-VL 推荐写法）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # 也可用 {"type": "image", "image": f"file://{os.path.abspath(image_path)}"}
                {"type": "text", "text": question},
            ],
        }
    ]

    # ① 生成文本模板（暂不tokenize）
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # ② 解析视觉输入（支持 PIL / 本地 file:// / URL / base64 等）
    image_inputs, video_inputs = process_vision_info(messages)
    # ③ 由 processor 将“文本+视觉”一并打包成张量
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print("正在生成回答...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # 仅解码新生成的部分
    new_ids = [o[len(i):] for i, o in zip(inputs.input_ids, output_ids)]
    resp = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    return resp

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-7B 单图片测试")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--question", type=str, required=True, help="询问问题")
    parser.add_argument("--model_path", type=str, default="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2.5-VL-32B-Instruct", help="模型路径或本地目录")
    args = parser.parse_args()

    try:
        model, processor = load_model(args.model_path)
        answer = process_image_and_text(model, processor, args.image, args.question)
        print("\n" + "=" * 50)
        print("问题:", args.question)
        print("模型回答:")
        print(answer)
        print("=" * 50)
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("使用示例:")
        print("python test_single_pic_qwen25vl.py --image /path/to/your.jpg --question '这张图片里有什么？'")
    else:
        main()
