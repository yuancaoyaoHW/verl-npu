# 数据准备与切分指南

本文档说明如何为 Cascade RL + OPD 多域训练准备数据。

## 概述

Cascade RL 训练分为 4 个阶段，每个阶段使用不同的数据集：

- **Stage 1 (Math)**: GSM8K + DeepMath-103K
- **Stage 2 (Science)**: SciKnowEval
- **Stage 3 (Code)**: LiveCodeBench
- **Stage 4 (Tool)**: ToolAlpaca

所有数据需要转换为 verl 的 parquet 格式，并切分 train/test 集。

## 1. 下载数据集

### 1.1 使用下载脚本

```bash
cd /workspace/verl-npu
bash scripts/download_datasets.sh
```

脚本会下载以下数据集到 `datasets/` 目录：

- **DeepMath-103K**: 103K 数学问题（约 5GB）
- **SciKnowEval**: 28K 科学问答（约 115MB）
- **LiveCodeBench**: 竞赛编程题目（约 4.5GB）
- **ToolAlpaca**: 3.9K 工具调用实例（GitHub 仓库）

### 1.2 手动下载

如果脚本下载失败，可以手动下载：

```bash
# DeepMath-103K
huggingface-cli download zwhe99/DeepMath-103K --repo-type dataset --local-dir datasets/DeepMath-103K

# SciKnowEval
huggingface-cli download hicai-zju/SciKnowEval --repo-type dataset --local-dir datasets/SciKnowEval

# LiveCodeBench
huggingface-cli download livecodebench/code_generation_lite --repo-type dataset --local-dir datasets/LiveCodeBench

# ToolAlpaca
git clone https://github.com/tangqiaoyu/ToolAlpaca.git datasets/ToolAlpaca
```

### 1.3 GSM8K

GSM8K 需要单独准备：

```bash
docker exec verl-vllm python3 -c "
from datasets import load_dataset
import os
os.makedirs('/root/data/gsm8k', exist_ok=True)
ds = load_dataset('openai/gsm8k', 'main', split='train')
ds.to_parquet('/root/data/gsm8k/train.parquet')
ds_test = load_dataset('openai/gsm8k', 'main', split='test')
ds_test.to_parquet('/root/data/gsm8k/test.parquet')
print(f'GSM8K train: {len(ds)}, test: {len(ds_test)}')
"
```

## 2. 转换数据集

使用 `convert_datasets.py` 将原始数据转换为 verl 格式：

```bash
docker exec verl-vllm python3 /root/scripts/convert_datasets.py \
  --datasets deepmath sciknow livecode toolalpaca \
  --data-dir /home/dataset \
  --no-merge
```

**参数说明：**

- `--datasets`: 要转换的数据集列表（gsm8k/deepmath/sciknow/livecode/toolalpaca）
- `--data-dir`: 数据集根目录
- `--no-merge`: 每个数据集单独输出（推荐）
- `--max-samples`: 限制每个数据集的样本数（用于测试）

**输出文件：**

```
deepmath_train.parquet      # 57,630 samples (difficulty >= 6.0)
sciknow_train.parquet       # 70,196 samples (v1 train)
livecode_train.parquet      # 834 samples
toolalpaca_train.parquet    # 4,255 samples
```

**数据格式：**

每个 parquet 文件包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `data_source` | str | 数据源标识（如 `math/deepmath`） |
| `prompt` | list[dict] | Chat 格式 prompt（含 system + user） |
| `ability` | str | 能力标签（math/science/code/tool_use） |
| `reward_model` | dict | `{"ground_truth": "...", "style": "rule\|code"}` |
| `extra_info` | dict | 域特定元数据 |

## 3. 切分 Train/Test 集

### 3.1 DeepMath（Stage 1）

```bash
docker exec verl-vllm python3 -c "
import pandas as pd

df = pd.read_parquet('/root/data/cascade/deepmath_train.parquet')
print(f'原始样本数: {len(df)}')

# 切分 500 条做 test
test = df.sample(n=500, random_state=42)
train = df.drop(test.index).reset_index(drop=True)
test = test.reset_index(drop=True)

# 保存
train.to_parquet('/root/data/cascade/math_train.parquet', index=False)
test.to_parquet('/root/data/cascade/math_test.parquet', index=False)

print(f'Train: {len(train)}, Test: {len(test)}')
"
```

### 3.2 GSM8K + DeepMath 合并（可选）

如果要用 GSM8K + DeepMath 合并训练：

```bash
docker exec verl-vllm python3 -c "
import pandas as pd

# 读取 GSM8K
gsm8k_train = pd.read_parquet('/root/data/gsm8k/train.parquet')
gsm8k_test = pd.read_parquet('/root/data/gsm8k/test.parquet')

# 读取 DeepMath
deepmath = pd.read_parquet('/root/data/cascade/deepmath_train.parquet')

# 合并 train
math_train = pd.concat([gsm8k_train, deepmath], ignore_index=True)
math_train.to_parquet('/root/data/cascade/math_train.parquet', index=False)

# GSM8K test 作为 math test
gsm8k_test.to_parquet('/root/data/cascade/math_test.parquet', index=False)

print(f'Math train: {len(math_train)} (GSM8K: {len(gsm8k_train)}, DeepMath: {len(deepmath)})')
print(f'Math test: {len(gsm8k_test)}')
"
```

### 3.3 SciKnowEval（Stage 2）

SciKnowEval 已经是 train 集，需要切分 test：

```bash
docker exec verl-vllm python3 -c "
import pandas as pd

df = pd.read_parquet('/root/data/cascade/sciknow_train.parquet')
print(f'原始样本数: {len(df)}')

# 切分 1000 条做 test
test = df.sample(n=1000, random_state=42)
train = df.drop(test.index).reset_index(drop=True)
test = test.reset_index(drop=True)

train.to_parquet('/root/data/cascade/science_train.parquet', index=False)
test.to_parquet('/root/data/cascade/science_test.parquet', index=False)

print(f'Train: {len(train)}, Test: {len(test)}')
"
```

### 3.4 LiveCodeBench（Stage 3）

LiveCodeBench 样本数较少，切分 100 条做 test：

```bash
docker exec verl-vllm python3 -c "
import pandas as pd

df = pd.read_parquet('/root/data/cascade/livecode_train.parquet')
print(f'原始样本数: {len(df)}')

# 切分 100 条做 test
test = df.sample(n=100, random_state=42)
train = df.drop(test.index).reset_index(drop=True)
test = test.reset_index(drop=True)

train.to_parquet('/root/data/cascade/code_train.parquet', index=False)
test.to_parquet('/root/data/cascade/code_test.parquet', index=False)

print(f'Train: {len(train)}, Test: {len(test)}')
"
```

### 3.5 ToolAlpaca（Stage 4）

ToolAlpaca 切分 500 条做 test：

```bash
docker exec verl-vllm python3 -c "
import pandas as pd

df = pd.read_parquet('/root/data/cascade/toolalpaca_train.parquet')
print(f'原始样本数: {len(df)}')

# 切分 500 条做 test
test = df.sample(n=500, random_state=42)
train = df.drop(test.index).reset_index(drop=True)
test = test.reset_index(drop=True)

train.to_parquet('/root/data/cascade/tool_train.parquet', index=False)
test.to_parquet('/root/data/cascade/tool_test.parquet', index=False)

print(f'Train: {len(train)}, Test: {len(test)}')
"
```

## 4. 验证数据

### 4.1 检查数据格式

```bash
docker exec verl-vllm python3 -c "
import pandas as pd

for name in ['math', 'science', 'code', 'tool']:
    train_path = f'/root/data/cascade/{name}_train.parquet'
    test_path = f'/root/data/cascade/{name}_test.parquet'
    
    try:
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        
        print(f'=== {name.upper()} ===')
        print(f'  Train: {len(train)}, Test: {len(test)}')
        print(f'  data_source: {train.iloc[0][\"data_source\"]}')
        print(f'  ability: {train.iloc[0][\"ability\"]}')
        
        # 检查 ground_truth
        gt = train.iloc[0]['reward_model']['ground_truth']
        print(f'  ground_truth sample: {str(gt)[:100]}')
        print()
    except Exception as e:
        print(f'  ERROR: {e}')
        print()
"
```

### 4.2 检查 reward 函数

```bash
docker exec verl-vllm python3 -c "
import sys
sys.path.insert(0, '/root/scripts')
from my_rewards import compute_score

# Math
r = compute_score('math/deepmath', '#### 42', '42')
print(f'Math correct: {r}')

r = compute_score('math/deepmath', '#### 43', '42')
print(f'Math wrong: {r}')

# Science
r = compute_score('science/sciknow', 'Answer: B', 'B', extra_info={'type': 'mcq-4-choices'})
print(f'Science MCQ: {r}')

# Code (no sandbox)
r = compute_score('code/livecode', '\`\`\`python\nprint(2+2)\n\`\`\`', '')
print(f'Code format: {r}')

# Tool
r = compute_score('tool/toolalpaca', 'Thought: test\nAction: search\nAction Input: {\"q\": \"test\"}', '[[\"search\", \"{\\\"q\\\": \\\"test\\\"}\"], \"obs\"]')
print(f'Tool: {r}')
"
```

## 5. 数据目录结构

最终的数据目录结构：

```
/root/data/cascade/
├── math_train.parquet       # Stage 1 train (GSM8K + DeepMath)
├── math_test.parquet        # Stage 1 test
├── science_train.parquet    # Stage 2 train (SciKnowEval)
├── science_test.parquet     # Stage 2 test
├── code_train.parquet       # Stage 3 train (LiveCodeBench)
├── code_test.parquet        # Stage 3 test
├── tool_train.parquet       # Stage 4 train (ToolAlpaca)
└── tool_test.parquet        # Stage 4 test
```

## 6. 常见问题

### Q: 数据集下载失败怎么办？

A: 使用 HuggingFace 镜像或手动下载后上传到服务器。

### Q: 转换脚本报错 "Cannot find data files"？

A: 检查数据集目录结构是否正确。DeepMath 应该在 `datasets/DeepMath-103K/data/` 下有 parquet 文件。

### Q: 如何只转换部分数据集？

A: 使用 `--max-samples` 参数限制样本数：

```bash
docker exec verl-vllm python3 /root/scripts/convert_datasets.py \
  --datasets deepmath \
  --data-dir /home/dataset \
  --max-samples 1000 \
  --no-merge
```

### Q: 如何验证 reward 函数是否正确？

A: 运行 reward 函数的自测：

```bash
docker exec verl-vllm python3 /root/scripts/my_rewards.py
```

应该输出 "✅ All tests passed!"。

## 7. 下一步

数据准备完成后，参考 [Cascade RL 训练指南](cascade-rl-training.md) 启动训练。
