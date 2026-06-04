# Haidass OPD on NPU — GSM8K On-Policy Distillation

基于 [verl PR #5837](https://github.com/verl-project/verl/pull/5837)，在 Ascend 910B3 NPU 上运行 Haidass (Qwen3-1.2B) ← Qwen3.6-35B-A3B 的 On-Policy Distillation 训练。

## 配置

| 项 | 值 |
|---|---|
| 学生模型 | Haidass (Qwen3-1.2B, 596M params) |
| 教师模型 | Qwen3.6-35B-A3B (Qwen3_5MoeForConditionalGeneration) |
| 数据集 | GSM8K (train: 7473, test: 1319) |
| 训练 | 15 epochs, 7005 steps, k1 KL + policy gradient |
| 硬件 | 8× Ascend 910B3 (64GB HBM each), 4 GPU student + 4 GPU teacher |
| 容器 | `quay.io/ascend/vllm-ascend:v0.18.0` |
| vLLM | 0.18.0 + vllm-ascend 0.18.0 |
| verl | 0.9.0.dev0 |
| transformers | 4.57.6 |

## 项目结构

```
haidass-opd-npu/
├── README.md
├── scripts/
│   ├── run_opd.sh                              # 训练启动脚本 (vLLM 0.18.0)
│   ├── run_haidass_qwen36_distill_gsm8k_npu.sh # 参考脚本 (vLLM 0.10.1)
│   └── run_qwen_haidass_distill_gsm8k_npu.sh   # 原始脚本
├── patches/
│   ├── teacher_model.py                        # verl patch: 注册 qwen3_5_moe config
│   └── apply-patch.sh                          # patch 应用脚本
├── models/
│   ├── Qwen3.6-35B-A3B/
│   │   ├── config.json                         # 教师模型 config
│   │   └── configuration_qwen3_5_moe.py        # 自定义 config 类
│   └── Haidass/
│       └── config.json                         # 学生模型 config
├── docs/
│   └── pr5837-opd-summary-final.md             # 完整排查文档
└── docker/
    └── docker-run.sh                           # 容器启动命令
```

## 快速开始

### 1. 启动容器

```bash
bash docker/docker-run.sh
```

### 2. 安装 verl 并应用 patch

```bash
docker exec verl-vllm bash -c "
  cd /tmp && git clone https://github.com/verl-project/verl.git && cd verl && pip install -e .
"
docker exec verl-vllm bash /path/to/patches/apply-patch.sh
```

### 3. 准备数据和模型

确保以下路径存在：
- `/root/data/gsm8k/train.parquet` — GSM8K 训练集
- `/root/data/gsm8k/test.parquet` — GSM8K 测试集
- `/home/models/Haidass/` — 学生模型（含 tokenizer 文件）
- `/home/models/Qwen3.6-35B-A3B/` — 教师模型（含 `configuration_qwen3_5_moe.py`）

### 4. 启动训练

```bash
docker exec -d verl-vllm bash -c "nohup bash /root/run_opd.sh > /home/admin/train_opd.log 2>&1 &"
```

### 5. 监控

```bash
# 查看日志
docker exec verl-vllm tail -50 /home/admin/train_opd.log

# 查看 NPU 状态
docker exec verl-vllm npu-smi info

# 查看 checkpoint
docker exec verl-vllm ls /tmp/verl/checkpoints/verl_opd_haidass_gsm8k/
```

## 已知问题与修复

详见 [docs/pr5837-opd-summary-final.md](docs/pr5837-opd-summary-final.md) 第五节。

三个关键修复：
1. **qwen3_5_moe 注册** — transformers 4.57.6 不认识该架构，需在 verl 代码中手动注册
2. **PYTORCH_NPU_ALLOC_CONF** — `expandable_segments:True` 与 vllm-ascend 0.18.0 的 CaMemAllocator 冲突，需移除
3. **transformers 版本** — 必须锁定 4.57.6，5.x 与 vLLM 0.18.0 不兼容
