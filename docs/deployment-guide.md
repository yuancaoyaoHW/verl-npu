# Cascade RL 部署指南

本文档说明如何在新的 NPU 服务器上部署 Cascade RL + OPD 多域训练环境。

## 概述

部署流程：
1. 启动容器（带正确的环境变量和 IPC 配置）
2. 初始化环境（verl + patches + 依赖）
3. 准备数据（下载 + 转换 + 切分）
4. 运行冒烟测试

## 1. 启动容器

### 1.1 使用 docker-run.sh（推荐）

```bash
cd /workspace/verl-npu
bash docker/docker-run.sh
```

脚本会自动创建 `verl-vllm` 容器，包含：
- `--ipc=host` - HCCL 多进程通信必需
- `-e NCCL_P2P_DISABLE=1` - 禁用 P2P（Ascend 910B 不支持）
- `-e HCCL_BUFFSIZE=300` - HCCL 缓冲区大小
- 8 个 NPU 设备映射
- `/home`、`/mnt`、`/tmp` 卷挂载

### 1.2 手动启动

如果需要自定义配置：

```bash
docker run -d --name verl-vllm \
  --network host --privileged --ipc=host --shm-size 32g \
  --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home:/home -v /mnt:/mnt -v /tmp:/tmp \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NCCL_P2P_DISABLE=1 \
  -e HCCL_BUFFSIZE=300 \
  quay.io/ascend/vllm-ascend:v0.18.0 \
  sleep infinity
```

### 1.3 验证容器配置

```bash
# 检查 IPC 模式
docker inspect verl-vllm --format '{{.HostConfig.IpcMode}}'
# 应输出: host

# 检查环境变量
docker exec verl-vllm env | grep -E 'NCCL|HCCL'
# 应输出:
# NCCL_P2P_DISABLE=1
# HCCL_BUFFSIZE=300

# 检查 NPU 设备
docker exec verl-vllm npu-smi info | head -20
```

## 2. 初始化环境

### 2.1 启动 Ray

```bash
docker exec verl-vllm bash -c "
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ray start --head --port=6379 --disable-usage-stats
"
```

### 2.2 安装 verl

```bash
docker cp /workspace/verl-npu/docker-build/verl-src verl-vllm:/tmp/verl
docker exec verl-vllm bash -c "cd /tmp/verl && pip install -e . --no-deps"
```

### 2.3 应用 patches

```bash
docker cp /workspace/verl-npu/patches verl-vllm:/opt/patches
docker exec verl-vllm bash /opt/patches/apply-patch.sh
```

### 2.4 复制训练脚本

```bash
docker cp /workspace/verl-npu/scripts verl-vllm:/root/scripts
```

### 2.5 安装依赖

```bash
docker exec verl-vllm pip install \
  'math-verify[antlr4_11_0]==0.7.0' \
  sympy \
  requests \
  tensordict \
  omegaconf \
  hydra-core \
  codetiming \
  datasets \
  torchdata \
  peft
```

### 2.6 验证环境

```bash
docker exec verl-vllm python3 -c "
import verl
import torch
import torch_npu
print(f'verl: {verl.__version__}')
print(f'torch: {torch.__version__}')
print(f'torch_npu: {torch_npu.__version__}')
print(f'NPU count: {torch_npu.npu.device_count()}')
"
```

## 3. 准备数据

详见 [数据准备指南](data-preparation-guide.md)。

### 3.1 快速准备（仅 DeepMath）

```bash
# 转换 DeepMath
docker exec verl-vllm bash -c "
  mkdir -p /root/data/cascade
  python3 /root/scripts/convert_datasets.py \
    --datasets deepmath \
    --data-dir /home/dataset \
    --no-merge
  cp /workspace/deepmath_train.parquet /root/data/cascade/math_train.parquet
"

# 切分 train/test
docker exec verl-vllm python3 -c "
import pandas as pd
df = pd.read_parquet('/root/data/cascade/math_train.parquet')
test = df.sample(n=500, random_state=42)
train = df.drop(test.index).reset_index(drop=True)
test = test.reset_index(drop=True)
train.to_parquet('/root/data/cascade/math_train.parquet', index=False)
test.to_parquet('/root/data/cascade/math_test.parquet', index=False)
print(f'Train: {len(train)}, Test: {len(test)}')
"
```

### 3.2 完整数据准备

```bash
# 转换所有数据集
docker exec verl-vllm python3 /root/scripts/convert_datasets.py \
  --datasets deepmath sciknow livecode toolalpaca \
  --data-dir /home/dataset \
  --no-merge

# 切分各域的 train/test（详见数据准备指南）
```

## 4. 运行冒烟测试

### 4.1 Stage 1 (Math) 冒烟测试

```bash
docker exec verl-vllm bash -c "
  nohup bash /root/scripts/run_cascade_stage1.sh \
    trainer.total_epochs=1 \
    trainer.test_freq=5 \
    trainer.save_freq=10 \
    data.train_batch_size=4 \
    data.train_files=/root/data/cascade/math_train.parquet \
    data.val_files=/root/data/cascade/math_test.parquet \
    actor_rollout_ref.rollout.n=8 \
    custom_reward_function.path=/root/scripts/my_rewards.py \
    > /home/admin/train_stage1_smoke.log 2>&1 &
"
```

### 4.2 监控日志

```bash
# 等待 90 秒让训练启动
sleep 90

# 查看日志
docker exec verl-vllm tail -100 /home/admin/train_stage1_smoke.log

# 实时监控
docker exec verl-vllm tail -f /home/admin/train_stage1_smoke.log
```

### 4.3 预期输出

成功启动后，日志应包含：

```
[INFO] Initializing RayPPOTrainer...
[INFO] Building actor_rollout_ref worker group...
[INFO] Building critic worker group...
[INFO] Building reward worker group...
[INFO] Starting training loop...
step_time: XX.Xs, throughput: XX.X tok/s
```

## 5. 常见问题

### Q: HCCL 通信失败（hcclCommInitRootInfoConfig error）

**原因**：Ray worker 进程没有继承环境变量。

**解决**：
1. 确保 `docker run` 包含 `-e NCCL_P2P_DISABLE=1` 和 `-e HCCL_BUFFSIZE=300`
2. 不要只在脚本里 `export`，必须写进 docker run
3. 重建容器：`docker stop verl-vllm && docker rm verl-vllm && bash docker/docker-run.sh`

### Q: IPC 模式不是 host

**原因**：容器启动时没有 `--ipc=host`。

**解决**：
```bash
docker inspect verl-vllm --format '{{.HostConfig.IpcMode}}'
# 如果不是 host，重建容器
docker stop verl-vllm && docker rm verl-vllm
bash docker/docker-run.sh
```

### Q: verl 模块找不到

**原因**：verl 没有正确安装。

**解决**：
```bash
docker exec verl-vllm bash -c "cd /tmp/verl && pip install -e . --no-deps"
docker exec verl-vllm python3 -c "import verl; print(verl.__file__)"
```

### Q: reward_model 模块找不到

**原因**：`my_rewards.py` 没有复制到容器里。

**解决**：
```bash
docker cp /workspace/verl-npu/scripts/my_rewards.py verl-vllm:/root/scripts/
```

### Q: 数据文件找不到

**原因**：数据路径不正确。

**解决**：
```bash
# 检查数据文件
docker exec verl-vllm ls -lh /root/data/cascade/

# 如果文件不存在，重新准备数据（见第 3 节）
```

### Q: NPU 显存不足

**原因**：batch size 太大或 rollout.n 太大。

**解决**：
```bash
# 减小 batch size
data.train_batch_size=2

# 减小 rollout.n
actor_rollout_ref.rollout.n=4

# 减小 response length
data.max_response_length=512
```

### Q: 训练卡住不动

**原因**：Ray worker 死锁或 NPU 僵尸进程。

**解决**：
```bash
# 杀掉训练进程
docker exec verl-vllm pkill -f "python3 -m verl"

# 重启 Ray
docker exec verl-vllm bash -c "ray stop --force; ray start --head --port=6379"

# 如果还是不行，重启容器
docker restart verl-vllm
sleep 10
docker exec verl-vllm bash -c "ray start --head --port=6379"
```

## 6. 完整部署脚本

一键部署脚本（假设 `/workspace/verl-npu` 已存在）：

```bash
#!/bin/bash
set -e

echo "=== Step 1: 启动容器 ==="
cd /workspace/verl-npu
docker stop verl-vllm 2>/dev/null || true
docker rm verl-vllm 2>/dev/null || true
bash docker/docker-run.sh

echo "=== Step 2: 初始化环境 ==="
docker exec verl-vllm bash -c "
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ray start --head --port=6379 --disable-usage-stats
"

docker cp /workspace/verl-npu/docker-build/verl-src verl-vllm:/tmp/verl
docker exec verl-vllm bash -c "cd /tmp/verl && pip install -e . --no-deps"

docker cp /workspace/verl-npu/patches verl-vllm:/opt/patches
docker exec verl-vllm bash /opt/patches/apply-patch.sh

docker cp /workspace/verl-npu/scripts verl-vllm:/root/scripts

docker exec verl-vllm pip install \
  'math-verify[antlr4_11_0]==0.7.0' \
  sympy requests tensordict omegaconf hydra-core \
  codetiming datasets torchdata peft

echo "=== Step 3: 准备数据 ==="
docker exec verl-vllm bash -c "
  mkdir -p /root/data/cascade
  python3 /root/scripts/convert_datasets.py \
    --datasets deepmath \
    --data-dir /home/dataset \
    --no-merge
  cp /workspace/deepmath_train.parquet /root/data/cascade/math_train.parquet
"

docker exec verl-vllm python3 -c "
import pandas as pd
df = pd.read_parquet('/root/data/cascade/math_train.parquet')
test = df.sample(n=500, random_state=42)
train = df.drop(test.index).reset_index(drop=True)
test = test.reset_index(drop=True)
train.to_parquet('/root/data/cascade/math_train.parquet', index=False)
test.to_parquet('/root/data/cascade/math_test.parquet', index=False)
print(f'Train: {len(train)}, Test: {len(test)}')
"

echo "=== Step 4: 冒烟测试 ==="
docker exec verl-vllm bash -c "
  nohup bash /root/scripts/run_cascade_stage1.sh \
    trainer.total_epochs=1 \
    trainer.test_freq=5 \
    trainer.save_freq=10 \
    data.train_batch_size=4 \
    data.train_files=/root/data/cascade/math_train.parquet \
    data.val_files=/root/data/cascade/math_test.parquet \
    actor_rollout_ref.rollout.n=8 \
    custom_reward_function.path=/root/scripts/my_rewards.py \
    > /home/admin/train_stage1_smoke.log 2>&1 &
"

echo "=== 部署完成 ==="
echo "监控日志: docker exec verl-vllm tail -f /home/admin/train_stage1_smoke.log"
```

## 7. 下一步

冒烟测试通过后：

1. **完整训练**：移除冒烟测试的限制参数，使用默认配置
2. **多域训练**：准备 Science/Code/Tool 数据，依次运行 Stage 2-4
3. **监控训练**：使用 `npu-smi info` 监控 NPU 利用率
4. **Checkpoint 管理**：定期备份 `/tmp/verl/checkpoints/`

详见 [Cascade RL 训练指南](cascade-rl-training.md)。
