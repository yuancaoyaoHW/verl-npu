# Haidass OPD NPU Training Image

基于 `quay.io/ascend/vllm-ascend:v0.18.0` 构建的 Haidass OPD 训练镜像。

## 镜像内容

- **基础镜像**: vLLM 0.18.0 + vllm-ascend 0.18.0 + torch 2.9.0 + torch_npu 2.9.0 + CANN 8.5.1
- **verl**: 0.9.0.dev0 (已安装，含 4 个 patches)
- **单域训练**: `/root/run_opd.sh` (Wave 5.1 配置, GSM8K OPD)
- **Cascade RL**: `/root/cascade/` (Math→Science→Code→Tool 多域训练)
- **Reward 依赖**: math-verify 0.7.0, sympy, requests

## Patches 已应用

1. `teacher_model.py` - qwen3_5_moe CONFIG_MAPPING 注册
2. `teacher_manager.py` - precompute_prompt_logprobs 方法
3. `agent_loop.py` - split teacher logprobs
4. `ray_trainer.py` - next-batch prefetch

## 使用方法

### 启动容器

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
  -e NCCL_P2P_DISABLE=1 \
  -e HCCL_BUFFSIZE=300 \
  haidass-opd-npu:v6.0
```

容器启动后会自动启动 Ray。

### 单域 OPD 训练 (GSM8K)

```bash
docker exec -d verl-vllm bash -c "nohup bash /root/run_opd.sh > /home/admin/train_opd.log 2>&1 &"
```

### Cascade RL 多域训练 (Math→Science→Code→Tool)

```bash
# 运行全部 4 个阶段
docker exec -d verl-vllm bash -c "nohup bash /root/cascade/run_all_stages.sh > /home/admin/train_cascade.log 2>&1 &"

# 或从指定阶段开始
docker exec -d verl-vllm bash -c "nohup bash /root/cascade/run_all_stages.sh 2 > /home/admin/train_cascade.log 2>&1 &"

# 或单独运行某个阶段
docker exec -d verl-vllm bash -c "nohup bash /root/cascade/run_cascade_stage1.sh > /home/admin/train_stage1.log 2>&1 &"
```

### 监控训练

```bash
docker exec verl-vllm tail -f /home/admin/train_opd.log
docker exec verl-vllm npu-smi info
docker exec verl-vllm ray status
```

## 配置对比

| 项 | 单域 OPD (run_opd.sh) | Cascade Stage 1 (Math) |
|---|---|---|
| 数据 | GSM8K (7.5K) | GSM8K + DeepMath (~64K) |
| rollout.n | 1 | 16 (GRPO) |
| batch_size | 48 | 12 |
| KL penalty | 无 | beta=0.01 |
| Reward | 无 (纯 OPD) | 双验证器 (string + math_verify) |
| LR | 1e-6 | 1e-6 |

## 重新构建

```bash
cd docker-build
docker build -t haidass-opd-npu:v6.0 .
```

## 文件结构

```
docker-build/
├── Dockerfile              # 构建定义
├── apply-patches.sh        # Patch 应用脚本
├── run_opd.sh              # 单域 OPD 训练脚本 (Wave 5.1)
├── cascade/                # Cascade RL 多域训练
│   ├── my_rewards.py       # 统一 reward 函数 (4 域路由)
│   ├── run_cascade_stage1.sh  # Math RL+OPD
│   ├── run_cascade_stage2.sh  # Science RL+OPD
│   ├── run_cascade_stage3.sh  # Code RL+OPD (需 SandboxFusion)
│   ├── run_cascade_stage4.sh  # Tool-Use RL+OPD
│   ├── run_all_stages.sh      # 总编排器
│   ├── convert_datasets.py    # 数据转换脚本
│   ├── download_datasets.sh   # 数据下载脚本
│   └── deploy_sandbox.sh      # SandboxFusion 部署
├── patches/                # 4 个 verl patches
│   ├── teacher_model.py
│   ├── teacher_manager.py
│   ├── agent_loop.py
│   └── ray_trainer.py
└── verl-src/               # verl 源码
```
