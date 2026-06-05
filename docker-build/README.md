# Haidass OPD NPU Training Image

基于 `quay.io/ascend/vllm-ascend:v0.18.0` 构建的 Haidass OPD 训练镜像。

## 镜像内容

- **基础镜像**: vLLM 0.18.0 + vllm-ascend 0.18.0 + torch 2.9.0 + torch_npu 2.9.0 + CANN 8.5.1
- **verl**: 0.9.0.dev0 (已安装，含 4 个 patches)
- **训练脚本**: `/root/run_opd.sh` (Wave 5.1 配置)

## Patches 已应用

1. `teacher_model.py` - qwen3_5_moe CONFIG_MAPPING 注册
2. `teacher_manager.py` - precompute_prompt_logprobs 方法
3. `agent_loop.py` - split teacher logprobs
4. `ray_trainer.py` - next-batch prefetch

## 使用方法

### 启动容器

```bash
docker run -d --name verl-vllm \
  --network host --privileged --shm-size 32g \
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
  haidass-opd-npu:v5.1
```

容器启动后会自动启动 Ray。

### 启动训练

```bash
docker exec -d verl-vllm bash -c "nohup bash /root/run_opd.sh > /home/admin/train_opd.log 2>&1 &"
```

### 监控训练

```bash
# 查看日志
docker exec verl-vllm tail -f /home/admin/train_opd.log

# 查看 NPU 状态
docker exec verl-vllm npu-smi info

# 查看 Ray 状态
docker exec verl-vllm ray status
```

## 当前配置 (Wave 5.1)

- **Student**: DP=4, TP=1, enforce_eager=False (graph capture), sleep_mode=False
- **Teacher**: TP=4, enforce_eager=False (graph capture)
- **Batch**: train_batch_size=48, micro_batch=16
- **性能**: step_time=29.8s, throughput=109 tok/s
- **预估总时间**: ~19h (2325 steps)

## 重新构建

```bash
cd docker-build
docker build -t haidass-opd-npu:v5.1 .
```

## 文件结构

```
docker-build/
├── Dockerfile              # 构建定义
├── apply-patches.sh        # Patch 应用脚本
├── run_opd.sh              # 训练启动脚本
├── patches/                # 4 个 verl patches
│   ├── teacher_model.py
│   ├── teacher_manager.py
│   ├── agent_loop.py
│   └── ray_trainer.py
└── verl-src/               # verl 源码 (23MB, 不含 checkpoints)
```
