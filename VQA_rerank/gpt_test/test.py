# filename: vision_two_images.py
import os
import base64
from openai import OpenAI
# test 

def encode_image_to_data_url(path: str) -> str:
    """把本地图片转成 data URL（无需对外可访问）"""
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def build_image_part(url_or_path: str) -> dict:
    """
    既支持 http(s) URL，也支持本地文件路径。
    OpenAI Chat Completions 的图片消息需要：
    {"type": "image_url", "image_url": {"url": "..."}}
    """
    if url_or_path.startswith("http://") or url_or_path.startswith("https://") or url_or_path.startswith("data:"):
        url = url_or_path
    else:
        url = encode_image_to_data_url(url_or_path)

    return {
        "type": "image_url",
        "image_url": {"url": url}
    }

def analyze_two_images(prompt: str, image1: str, image2: str, model: str = "gpt-4o") -> str:
    """
    prompt: 你的文字说明/问题
    image1, image2: 本地路径或 URL
    model: 需为支持视觉输入的模型（如 gpt-4o / gpt-4o-mini）
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                build_image_part(image1),
                build_image_part(image2),
            ],
        }
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    # ====== 在这里自定义你的 prompt 与两张图片 ======
    user_prompt = "Please describe the difference between the two images."
    img_a = "/home/guohaiyun/yty_data/CIRR/dev/dev-99-2-img0.png"                     # 本地文件
    img_b = "/home/guohaiyun/yty_data/CIRR/dev/dev-414-1-img0.png" # 或者 URL（两者都支持）

    print(analyze_two_images(user_prompt, img_a, img_b))
