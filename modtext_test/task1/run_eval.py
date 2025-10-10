#!/usr/bin/env python3
"""
运行CIRR多图理解能力测试的示例脚本
"""

import subprocess
import sys
import os

def run_test():
    """运行测试"""
    print("CIRR多图理解能力测试 - 运行脚本")
    print("=" * 50)
    
    # 基本参数
    cmd = [
        sys.executable, "eval_found.py",
        "--num_groups", "3",  # 测试3个组
        "--max_pairs_per_group", "2",  # 每组最多2对图片
        "--model_name", "/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct",
        "--device", "auto"
    ]
    
    print("运行命令:")
    print(" ".join(cmd))
    print("\n开始执行...")
    print("注意：每对图片将使用两种方法进行比较：")
    print("  方法1：直接询问差异")
    print("  方法2：先生成caption再询问差异")
    print()
    
    try:
        # 运行测试
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("测试完成!")
        print("\n标准输出:")
        print(result.stdout)
        
        if result.stderr:
            print("\n标准错误:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"测试失败，退出码: {e.returncode}")
        print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"运行时错误: {e}")
        return False
    
    return True

def show_usage():
    """显示使用说明"""
    print("""
CIRR多图理解能力测试工具使用说明 (更新版)
=========================================

主要功能：
- 从CIRR数据集中读取图片组
- 对组内图片进行两两比较，使用两种不同方法
- 使用Qwen2VL-7B模型分析图片差异
- 保存所有比较结果和图片ID，文件名包含时间戳

比较方法：
方法1 - 直接比较：直接询问两张图片的差异
方法2 - 基于描述：先为每张图片生成caption，再基于caption比较差异

主要参数：
--num_groups         测试的组数量 (默认: 10)
--max_pairs_per_group 每组内最大测试的图片对数量 (默认: 5)
--output_file        结果保存文件路径 (自动添加时间戳)
--model_name         Qwen2VL模型名称或路径
--device             设备选择 (auto/cuda/cpu)
--cirr_data_dir      CIRR数据集目录路径
--cirr_image_dir     CIRR图片基础目录路径

示例用法：
1. 快速测试 (3个组，每组2对图片):
   python eval_found.py --num_groups 3 --max_pairs_per_group 2

2. 中等规模测试 (20个组，每组5对图片):
   python eval_found.py --num_groups 20 --max_pairs_per_group 5

3. 指定输出文件前缀:
   python eval_found.py --output_file my_test.json
   # 实际文件名: my_test_20240101_123456.json

4. 使用本地模型:
   python eval_found.py --model_name /path/to/local/qwen2vl/model

输出格式：
结果保存为JSON格式，包含：
- metadata: 测试元信息（时间、参数、方法说明等）
- results: 每个图片对比较的详细结果
  - group_name: 组名
  - image1_id/image2_id: 两张图片的ID
  - comparison_methods: 包含两种方法的结果
    - method1_direct: 直接比较的结果和处理时间
    - method2_caption_based: 基于caption的比较结果、两个caption和处理时间
  - total_processing_time_seconds: 总处理时间
  - timestamp: 时间戳

方法比较示例：
方法1输出: "The main differences between these two images are..."
方法2输出: 
  - image1_caption: "This image shows a living room with..."
  - image2_caption: "This image displays a bedroom with..."
  - comparison_result: "Based on these descriptions, the main differences are..."

注意事项：
- 方法2需要3次模型调用，处理时间较长
- 确保CIRR数据集路径正确
- 首次运行会加载Qwen2VL模型
- 所有交互均使用英文
- 文件名自动包含时间戳避免覆盖
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_usage()
    else:
        success = run_test()
        if success:
            print("\n✓ 测试运行完成!")
        else:
            print("\n✗ 测试运行失败!")
            sys.exit(1) 