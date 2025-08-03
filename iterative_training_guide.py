#!/usr/bin/env python3
"""
Iterative Training Resume Examples

This script demonstrates how to use the improved iterative training with model loading
handled at the entry point using MMEBModel.build() and MMEBModel.load().
"""

import os

def show_resume_examples():
    """Show different ways to resume iterative training"""
    
    print("ITERATIVE TRAINING RESUME EXAMPLES")
    print("=" * 60)
    
    print("\n1. 🚀 AUTO RESUME (自动恢复):")
    print("   自动检测最新的迭代检查点并恢复")
    print("   命令: python train_iterative.py --resume_from auto [其他参数]")
    print("   行为: 自动找到最新的 iteration_X 检查点并从下一个迭代开始")
    
    print("\n2. 📍 MANUAL ITERATION RESUME (手动指定迭代恢复):")
    print("   手动指定从特定迭代恢复")
    print("   命令: python train_iterative.py --resume_from iter_2 [其他参数]")
    print("   行为: 加载 iteration_2 的模型权重，从第3次迭代开始训练")
    
    print("\n3. 🔄 STANDARD CHECKPOINT RESUME (标准检查点恢复):")
    print("   使用标准的HuggingFace检查点恢复")
    print("   命令: python train_iterative.py --resume_from checkpoint-1000 [其他参数]")
    print("   行为: 从 checkpoint-1000 恢复，但作为新的迭代训练开始")
    
    print("\n4. 🆕 FRESH START (全新开始):")
    print("   从头开始新的迭代训练")
    print("   命令: python train_iterative.py [其他参数]")
    print("   行为: 构建新模型，从第0次迭代开始")


def show_file_structure():
    """Show the file structure created by iterative training"""
    
    print("\n\nFILE STRUCTURE (文件结构)")
    print("=" * 60)
    
    structure = """
output_dir/
├── 📁 base_model/                      # 第0次迭代的基础模型
├── 📁 iteration_1/                     # 第1次迭代的模型权重
├── 📁 iteration_2/                     # 第2次迭代的模型权重  ⭐ 恢复点
├── 📁 iteration_3/                     # 第3次迭代的模型权重
├── 📄 iteration_0_state.json           # 第0次迭代的状态信息
├── 📄 iteration_1_state.json           # 第1次迭代的状态信息
├── 📄 iteration_2_state.json           # 第2次迭代的状态信息  ⭐ 恢复点
├── 📄 hard_negatives_iter_0.json       # 第0次迭代的困难负样本
├── 📄 hard_negatives_iter_1.json       # 第1次迭代的困难负样本
├── 📄 hard_negatives_iter_2.json       # 第2次迭代的困难负样本
├── 📄 augmented_samples_iter_1.json    # 第1次迭代的增强样本
├── 📄 augmented_samples_iter_2.json    # 第2次迭代的增强样本
├── 📄 augmented_samples_iter_3.json    # 第3次迭代的增强样本
└── 📄 training_summary.json            # 训练总结
    """
    
    print(structure)
    
    print("🔍 恢复机制:")
    print("1. MMEBModel.load() 加载模型权重 (LoRA/Full)")
    print("2. iteration_X_state.json 加载训练状态")
    print("3. augmented_samples_iter_X.json 恢复数据集状态")
    print("4. hard_negatives_iter_X.json 恢复困难负样本")


def show_advantages():
    """Show advantages of the new architecture"""
    
    print("\n\nARCHITECTURE ADVANTAGES (架构优势)")
    print("=" * 60)
    
    advantages = [
        ("🏗️  统一模型管理", "使用 MMEBModel.build() 和 MMEBModel.load() 统一管理模型加载"),
        ("🔄 LoRA支持", "自动处理LoRA适配器的加载和保存"),
        ("📊 职责分离", "入口文件负责初始化，trainer负责训练逻辑"),
        ("🛡️  错误处理", "模型加载失败时自动降级到基础模型"),
        ("⚡ 灵活恢复", "支持自动检测、手动指定、标准检查点等多种恢复方式"),
        ("🧹 代码简洁", "trainer不再处理复杂的模型加载逻辑"),
        ("🔧 易于维护", "与现有的VLM2Vec架构保持一致")
    ]
    
    for title, desc in advantages:
        print(f"{title}: {desc}")


def create_example_commands():
    """Create example command templates"""
    
    print("\n\nEXAMPLE COMMANDS (示例命令)")
    print("=" * 60)
    
    commands = {
        "自动恢复": """
python train_iterative.py \\
    --model_name Qwen/Qwen2-VL-2B-Instruct \\
    --output_dir ./experiments/iterative_cirr \\
    --resume_from auto \\
    --dataset_config configs/cirr_iterative.yaml \\
    --max_iterations 5
        """,
        
        "从第2次迭代恢复": """
python train_iterative.py \\
    --model_name Qwen/Qwen2-VL-2B-Instruct \\
    --output_dir ./experiments/iterative_cirr \\
    --resume_from iter_2 \\
    --dataset_config configs/cirr_iterative.yaml \\
    --max_iterations 5
        """,
        
        "全新开始": """
python train_iterative.py \\
    --model_name Qwen/Qwen2-VL-2B-Instruct \\
    --output_dir ./experiments/iterative_cirr_new \\
    --dataset_config configs/cirr_iterative.yaml \\
    --foundation_model_name Qwen/Qwen2-VL-7B-Instruct \\
    --max_iterations 3
        """
    }
    
    for scenario, command in commands.items():
        print(f"\n📝 {scenario}:")
        print(command.strip())


if __name__ == "__main__":
    show_resume_examples()
    show_file_structure()
    show_advantages()
    create_example_commands()
    
    print(f"\n\n{'='*60}")
    print("SUMMARY (总结)")
    print("="*60)
    print("✅ 模型加载现在使用 MMEBModel.load() 统一处理")
    print("✅ 支持自动和手动的迭代恢复")
    print("✅ LoRA权重自动管理")
    print("✅ 更清晰的架构分离")
    print("✅ 与现有VLM2Vec框架保持一致")
