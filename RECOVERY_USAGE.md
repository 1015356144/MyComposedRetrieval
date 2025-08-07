# 恢复机制使用指南

## 概述

新的恢复系统将两种不同的检查点类型完全分离：

1. **Trainer Checkpoints** (`--resume_from`): 包含完整训练状态（模型权重 + optimizer + scheduler）
2. **Iteration Models** (`--resume_from_iteration`): 只包含模型权重，用于迭代训练的不同阶段

## 参数说明

### --resume_from
控制从Trainer自动保存的检查点恢复，包含完整的训练状态。

- `auto`: 自动查找最新的trainer checkpoint
- `数字`: 指定特定的checkpoint步骤 (如 `20` 对应 `checkpoint-20`)
- `none`: 不使用trainer checkpoint (默认)

### --resume_from_iteration
控制从迭代训练保存的模型恢复，只包含模型权重。

- `auto`: 自动查找最新的完整迭代模型
- `iter_N`: 指定特定的迭代模型 (如 `iter_0`, `iter_1`, `iter_2`)
- `none`: 不使用迭代模型 (默认)

## 使用场景

### 场景1: 从头开始训练
```bash
# 两个参数都保持默认值
--resume_from none --resume_from_iteration none
```

### 场景2: 恢复标准训练 (使用trainer checkpoint)
```bash
# 恢复最新的trainer checkpoint (包含optimizer/scheduler状态)
--resume_from auto --resume_from_iteration none
```

### 场景3: 恢复迭代训练 (使用iteration model)
```bash
# 恢复最新的完整迭代，但训练状态从头开始
--resume_from none --resume_from_iteration auto
```

### 场景4: 恢复迭代训练 (同时使用两种checkpoint)
```bash
# 模型权重从迭代模型加载，训练状态从trainer checkpoint加载
--resume_from auto --resume_from_iteration auto
```

### 场景5: 指定特定的迭代和检查点
```bash
# 从第2次迭代的模型开始，使用第100步的训练状态
--resume_from 100 --resume_from_iteration iter_2
```

## 恢复优先级

系统会根据两个参数的组合自动选择最佳恢复策略：

1. **两个都有**: 使用iteration model权重 + trainer checkpoint训练状态
2. **只有trainer**: 完整恢复trainer checkpoint
3. **只有iteration**: 使用iteration model权重，训练状态从头开始
4. **都没有**: 从头开始训练

## 命令示例

### 全新开始迭代训练
```bash
torchrun --nproc_per_node=4 --master_port=29500 \
    train_iterative.py \
    --model_name "/path/to/model" \
    --output_dir "./experiments/new_experiment" \
    --resume_from none \
    --resume_from_iteration none \
    # ... 其他参数
```

### 恢复已有的迭代训练实验
```bash
torchrun --nproc_per_node=4 --master_port=29500 \
    train_iterative.py \
    --model_name "/path/to/model" \
    --output_dir "./experiments/existing_experiment" \
    --resume_from auto \
    --resume_from_iteration auto \
    # ... 其他参数
```

### 从特定迭代继续训练
```bash
torchrun --nproc_per_node=4 --master_port=29500 \
    train_iterative.py \
    --model_name "/path/to/model" \
    --output_dir "./experiments/existing_experiment" \
    --resume_from none \
    --resume_from_iteration iter_1 \
    # ... 其他参数
```

## 系统日志

新系统会在启动时显示详细的恢复信息：

```
============================================================
CHECKPOINT RECOVERY SYSTEM
============================================================
📁 Found trainer checkpoint: ./experiments/exp/checkpoint-20
   ✅ Contains: model weights + optimizer + scheduler states
🎯 Found COMPLETE iteration 0 model: ./experiments/exp/base_model
   ⚠️  Contains: model weights only (no optimizer/scheduler)
------------------------------------------------------------
RECOVERY STRATEGY:
🔀 BOTH checkpoints found - using ITERATION model for weights
   📋 Reason: Iteration models contain the latest trained weights
   🎯 Model weights from: ./experiments/exp/base_model
   📁 Training state from: ./experiments/exp/checkpoint-20
============================================================
```

## 注意事项

1. **Config文件**: 系统会自动为iteration models创建缺失的`config.json`
2. **训练状态**: 只有trainer checkpoints包含optimizer和scheduler状态
3. **优先级**: 当两种checkpoint都存在时，优先使用iteration model的权重
4. **错误处理**: 如果加载失败，系统会自动回退到其他可用选项
