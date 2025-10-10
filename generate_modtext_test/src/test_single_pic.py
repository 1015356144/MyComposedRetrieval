import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import argparse
import os

def _pick_dtype():
    # 优先 bfloat16（更稳），否则 float16；CPU 则回落 float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    return torch.float32

def load_model(model_path="Qwen/Qwen2-VL-7B-Instruct"):
    """加载 Qwen2-VL-7B 模型与处理器（支持单图/多图/视频）"""
    print("正在加载模型...")
    dtype = _pick_dtype()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,  # 环境支持则自动加速
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    print("模型加载完成!")
    return model, processor

def process_image_and_text(model, processor, image_path, question):
    """处理单张图片与文本输入，生成模型回答"""

    # 1) 检查与加载图片
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        print(f"图片加载成功: {image_path}")
        print(f"图片尺寸: {image.size}")
    except Exception as e:
        raise Exception(f"无法加载图片: {e}")

    # 2) 构建对话
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},      # 也可改成 {"type": "image", "path": image_path}
                {"type": "text", "text": question},
            ],
        }
    ]

    # 3) 由处理器统一打包“图像+文本”为张量输入
    #    关键点：tokenize=True + return_tensors="pt"
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    print("正在生成回答...")

    # 4) 生成
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.9,
            top_p=0.8,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # 5) 只解码新增部分
    new_token_ids = [o[len(i):] for i, o in zip(inputs["input_ids"], output_ids)]
    response = processor.batch_decode(new_token_ids, skip_special_tokens=True)[0]
    return response

def main():
    """主函数"""

    image="/home/guohaiyun/yty_data/CIRR/train/8/train-5793-0-img1.png"
    question="""Please think step by step using the follow steps: 
    ##Step1:
    please answer these questions. give the detailed reason first, then answer 'Yes' or 'No', the question order should be the same as below.
    - Your answer must only refer to the picture, do not use any other information.
    Is the animal a Pomeranian?
    Is the animal sitting?
    Is the surface white? 
    ##Step2:
    And then, here is a incorrect modification text: 'Pomeranian is sitting on a white surface instead of a gray one.'
    please using the answer of step1 to rewrite the modification text to match picture, give the new text: 
    """
    question2="Describe the breed of the animal in the picture."
    model_path="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2.5-VL-7B-Instruct"
    try: 
        model, processor = load_model(model_path)
        response = process_image_and_text(model, processor, image, question2)

        print("\n" + "=" * 50)
        print("问题:", question)
        print("模型回答:")
        print(response)
        print("=" * 50)

    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    import sys
    main()
