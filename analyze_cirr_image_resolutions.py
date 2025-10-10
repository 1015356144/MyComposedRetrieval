import json
import os
from PIL import Image
from pathlib import Path
import numpy as np
from collections import defaultdict

def analyze_cirr_image_resolutions(cirr_root="/home/guohaiyun/yty_data/CIRR", 
                                  hard_negatives_file="./experiments/IterativeCIRR_qwen2vl_20250829_005513_copy/hard_negatives_iter_0.json"):
    """分析CIRR数据集中图像的分辨率分布"""
    
    print("🔍 Analyzing CIRR image resolutions...")
    
    # 收集所有图像路径
    image_paths = set()
    
    # 从硬负样本文件中收集图像路径
    if os.path.exists(hard_negatives_file):
        with open(hard_negatives_file, 'r') as f:
            hard_negs = json.load(f)
        
        for item in hard_negs:
            image_paths.add(item['reference_image'])
            image_paths.add(item['target_image'])
            image_paths.add(item['hard_negative_image'])
    
    # 分析分辨率
    resolutions = []
    token_counts = []  # 基于Qwen2VL的token计算
    large_images = []  # 记录大分辨率图像
    
    for img_path in image_paths:
        full_path = os.path.join(cirr_root, img_path)
        if os.path.exists(full_path):
            try:
                with Image.open(full_path) as img:
                    width, height = img.size
                    total_pixels = width * height
                    resolutions.append((width, height, total_pixels))
                    
                    # Qwen2VL的token数量估算（基于14x14 patches + 额外token）
                    # 图像会被resize然后分patch
                    patch_size = 14
                    patches_w = (width + patch_size - 1) // patch_size
                    patches_h = (height + patch_size - 1) // patch_size
                    visual_tokens = patches_w * patches_h + 2  # +2 for special tokens
                    token_counts.append(visual_tokens)
                    
                    # 标记大图像（>1M像素或>400个tokens）
                    if total_pixels > 1000000 or visual_tokens > 400:
                        large_images.append({
                            'path': img_path,
                            'resolution': f"{width}x{height}",
                            'pixels': total_pixels,
                            'estimated_tokens': visual_tokens
                        })
                        
            except Exception as e:
                print(f"Error processing {full_path}: {e}")
    
    # 统计分析
    if resolutions:
        widths = [r[0] for r in resolutions]
        heights = [r[1] for r in resolutions]
        pixels = [r[2] for r in resolutions]
        
        print(f"\n📊 Resolution Analysis ({len(resolutions)} images):")
        print(f"Width  - Min: {min(widths):4d}, Max: {max(widths):4d}, Avg: {np.mean(widths):6.1f}")
        print(f"Height - Min: {min(heights):4d}, Max: {max(heights):4d}, Avg: {np.mean(heights):6.1f}")
        print(f"Pixels - Min: {min(pixels):7d}, Max: {max(pixels):7d}, Avg: {np.mean(pixels):9.0f}")
        
        print(f"\n🎯 Visual Token Analysis:")
        print(f"Tokens - Min: {min(token_counts):3d}, Max: {max(token_counts):3d}, Avg: {np.mean(token_counts):6.1f}")
        
        # 分辨率分布
        pixel_ranges = [
            (0, 100000, "Very Small"),
            (100000, 500000, "Small"), 
            (500000, 1000000, "Medium"),
            (1000000, 2000000, "Large"),
            (2000000, float('inf'), "Very Large")
        ]
        
        print(f"\n📈 Resolution Distribution:")
        for min_p, max_p, label in pixel_ranges:
            count = sum(1 for p in pixels if min_p <= p < max_p)
            percentage = count / len(pixels) * 100
            print(f"{label:12} ({min_p//1000:4d}K-{max_p//1000 if max_p != float('inf') else '∞':>4}K): {count:3d} images ({percentage:5.1f}%)")
        
        # 显示大图像
        if large_images:
            print(f"\n⚠️  Large Images (>1M pixels or >400 tokens): {len(large_images)} images")
            large_images.sort(key=lambda x: x['estimated_tokens'], reverse=True)
            print("Top 10 largest images:")
            for i, img in enumerate(large_images[:10], 1):
                print(f"{i:2d}. {img['path']} - {img['resolution']} ({img['estimated_tokens']} tokens)")
    
    return large_images

if __name__ == "__main__":
    large_images = analyze_cirr_image_resolutions()