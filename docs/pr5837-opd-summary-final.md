# PR #5837 — Haidass OPD 训练任务 改动/进度/待办汇总

**生成时间:** 2026-06-04 16:25
**最后更新:** 2026-06-04 09:45（训练已成功启动）
**原始任务:** 基于 [verl PR #5837](https://github.com/verl-project/verl/pull/5837)，在 Haidass (Qwen3-1.2B) + Qwen3.6-35B-A3B 教师模型上跑 GSM8K On-Policy Distillation，15 epochs，k1 + policy gradient。

---

## 一、PR #5837 分析摘要

PR 新增 `run_qwen_gsm8k_npu.sh`（学生 Qwen2.5-0.5B / 教师 Qwen2.5-3B-Instruct），与我们现有脚本差异：

| 维度 | PR 脚本 | 现有 Haidass 脚本 |
|---|---|---|
| 配置格式 | 旧版 `distillation.teacher_model.*` | 新版 `distillation.teacher_models.teacher_model.*` |
| `use_fused_kernels` | `True` | 未设置（默认 False） |
| `use_legacy_worker_impl` | `disable` | 不存在（当前代码已移除此选项） |
| `resume_mode` | `disable` | 未设置（默认 auto） |
| `log_val_generations` | `5` | 未设置（默认 0） |
| `val_before_train` | `True` | `False` |

Review 发现的 bug（**不适用于我们新版配置格式**）：
- 6-GPU 下 `TEACHER_RESOURCE_POOL=False` 导致 Ray 死锁
- `max_num_batched_tokens=769` 太低

---

## 二、已完成的改动

### 2.1 新脚本：`run_haidass_qwen36_distill_gsm8k_npu.sh`

从 `run_qwen_haidass_distill_gsm8k_npu.sh` 复制，diff 如下：

```diff
+ data.shuffle=False                         # 提高可复现性
+ actor_rollout_ref.model.use_fused_kernels=True  # fused kernels
+ trainer.resume_mode=disable               # 避免意外恢复旧 checkpoint
+ trainer.log_val_generations=5             # validation 打印 5 条生成样本
+ EXP_NAME 后缀 _v2                         # 区分新旧实验
+ tokenizer monkey-patch (见 3.3)           # 仅 vLLM 0.10 需要
```

文件路径：`/home/verl_latest/examples/on_policy_distillation_trainer/run_haidass_qwen36_distill_gsm8k_npu.sh`

### 2.2 Haidass 模型缺少 tokenizer → 已修复

`/home/models/Haidass/` 只有 `config.json` + `safetensors`，缺 tokenizer 文件 → verl 从 `model.path` 加载 tokenizer 时 `chat_template=None` → 全部 7473 条数据被过滤。

**修复:** 从 `/home/train/Qwen3-0.6B/` 复制 `tokenizer_config.json`、`tokenizer.json`、`vocab.json` 到 `/home/models/Haidass/`。

---

## 三、兼容性问题排查（完整历程）

### 3.1 教师模型 Qwen3.6-35B-A3B

架构 `Qwen3_5MoeForConditionalGeneration`，原生不支持 vLLM 0.10.1rc2。

**方案 A — Patch config.json → ❌**
- 改 `model_type=qwen3_moe`、`architectures=Qwen3MoeForCausalLM`
- vLLM 接受 config，但权重加载 assertion 失败

**方案 B — verl 容器内升级 vLLM 0.17.0 → ❌**
- `pip install vllm==0.17.0` 部分覆盖了镜像内置的 0.10.1rc2
- `EngineArgs` API 完全不同
- vllm-ascend 0.18.0 与 vLLM 0.17.0 不匹配
- triton 3.7.0 在 NPU 上报 `libcuda.so` 缺失
- 结论：verl 容器的 vLLM 环境已被破坏

**方案 C — 独立 vllm-ascend:v0.18.0 容器 ✅**
- 镜像：`quay.io/ascend/vllm-ascend:v0.18.0`
- vLLM 0.18.0 + vllm-ascend 0.18.0 + torch 2.9.0 + torch_npu 2.9.0
- 原生支持 `Qwen3_5MoeForConditionalGeneration`
- NPU 可见性调试（见 3.2）

### 3.2 vllm-ascend:v0.18.0 容器 NPU 适配

| 尝试 | 问题 | 修复 |
|---|---|---|
| 1 | NPU=0 | 挂载 `/usr/local/Ascend/driver` |
| 2 | NPU=0 | 挂载 `/usr/local/Ascend/driver/lib64` |
| 3 | NPU=0，`npu-smi: not found` | 挂载 `/usr/local/bin/npu-smi` |
| 4 | `libhccl.so` 找不到 | 不挂载宿主 CANN，让镜像用内置 CANN 8.5.1 |
| **5** | **NPU=8 ✅** | 所有挂载正确 |

**最终启动命令：**
```bash
docker run -d --name verl-vllm \
  --network host --privileged --shm-size 32g \
  --device=/dev/davinci0..7 --device=/dev/davinci_manager \
  --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home:/home -v /mnt:/mnt -v /tmp:/tmp \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  quay.io/ascend/vllm-ascend:v0.18.0 \
  sleep infinity
```

### 3.3 vLLM 0.10.1 专属 tokenizer monkey-patch

仅 vLLM 0.10.1 V1 引擎需要（transformers 5.x 移除了 `all_special_tokens_extended`）。vLLM 0.18.0 **不需要此 patch**。

---

## 四、当前容器环境

| 容器 | vLLM | NPU | verl | 状态 |
|---|---|---|---|---|
| **verl-vllm** | **0.18.0** | **8** | **0.9.0.dev0** | ✅ **目标容器** |
| verl-fresh | 0.10.1rc2 | 8 | 内置 | ⚠️ 备份，不支持 Qwen3.5MoE |
| verl (原始) | 0.17.0 (污染) | 8 | 内置 | ❌ 环境已破坏 |
| verl-vlm-grpo | 0.13.0 | — | 0.7.1 | 无关 |

**verl-vllm 就绪清单：**

| 项目 | 状态 |
|---|---|
| vLLM 0.18.0 原生支持 Qwen3_5MoeForConditionalGeneration | ✅ |
| 8 NPU 可用 | ✅ |
| verl 已安装 | ✅ |
| GSM8K 数据 `/root/data/gsm8k/train.parquet` | ✅ |
| 教师模型 config 为原始值（无 patch） | ✅ |
| Haidass tokenizer 文件完整 | ✅ |

---

## 五、训练启动问题修复（2026-06-04 续）

### 5.1 `/root/run_opd.sh` 已存在

文档之前标记为丢失，但实际已存在且内容正确（已适配 vLLM 0.18.0）。

### 5.2 训练启动遇到的 3 个 bug 及修复

#### Bug 1: `qwen3_5_moe` 架构不被 transformers 4.57.6 识别

**错误:** `ValueError: The checkpoint you are trying to load has model type 'qwen3_5_moe' but Transformers does not recognize this architecture.`

**根因:** verl 的 `HFModelConfig` 用 `AutoConfig.from_pretrained()` 加载教师模型 config，但 transformers 4.57.6 的 `CONFIG_MAPPING` 中没有 `qwen3_5_moe`。教师模型目录有自定义 `configuration_qwen3_5_moe.py`，但 `trust_remote_code=True` + `auto_map` 方案在 Ray 多进程下不可行（动态模块无法 pickle 序列化到 worker 进程）。

**修复:** 在 `/tmp/verl/verl/experimental/teacher_loop/teacher_model.py` 头部注入注册代码：

```python
# --- [verl-patch] Register custom Qwen3.5 MoE config ---
import importlib.util as _ilu
_teacher_cfg_file = '/home/models/Qwen3.6-35B-A3B/configuration_qwen3_5_moe.py'
if os.path.exists(_teacher_cfg_file):
    _sp = _ilu.spec_from_file_location('_qwen35moe_cfg', _teacher_cfg_file)
    _m = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if 'qwen3_5_moe' not in CONFIG_MAPPING:
        CONFIG_MAPPING.register('qwen3_5_moe', _m.Qwen3_5MoeConfig)
# --- [verl-patch] end ---
```

#### Bug 2: `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 与 vllm-ascend 0.18.0 不兼容

**错误:** `AssertionError: Expandable segments are not compatible with memory pool.`

**根因:** vllm-ascend 0.18.0 使用 `CaMemAllocator`（NPU 专用显存分配器），它断言不能与 `expandable_segments` 共存。PR #5837 的脚本是给老版 vLLM 0.10.1 写的，没有这个问题。

**修复:** 从 `/root/run_opd.sh` 中删除 `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`。

#### Bug 3: transformers 被意外升级到 5.10.1 导致 vLLM 崩溃

**错误:** `ModuleNotFoundError: No module named 'transformers.image_transforms'` / `No module named 'transformers.models.auto.modeling_auto'`

**根因:** 排查 Bug 1 时执行了 `pip install --upgrade transformers`（超时但部分完成），将 transformers 从 4.57.6 升级到 5.10.1。5.x 版本重组了内部模块结构，与 vLLM 0.18.0 不兼容。

**修复:**
```bash
pip install 'transformers==4.57.6' --no-deps -i https://mirrors.aliyun.com/pypi/simple/
pip install 'huggingface-hub>=0.34.0,<1.0' -i https://mirrors.aliyun.com/pypi/simple/
```

### 5.3 ✅ 训练已成功启动

- **时间:** 2026-06-04 09:42
- **日志:** `/home/admin/train_opd_vllm018.log`
- **状态:** `Training Progress: 0%| | 1/7005 [01:39<194:05:19, 99.76s/it]`
- **教师模型:** Qwen3.6-35B-A3B vLLM (TP=4) ✅
- **学生模型:** Haidass vLLM (TP=1) + FSDP (4 GPU) ✅
- **预计总时间:** ~194 小时（约 8 天）

---

## 六、未完成的任务

### 6.1 🟡 监控训练进度

训练已在后台运行，需定期检查：
- 日志：`docker exec verl-vllm tail -50 /home/admin/train_opd_vllm018.log`
- NPU 利用率：`docker exec verl-vllm npu-smi info`
- Checkpoint：`docker exec verl-vllm ls /tmp/verl/checkpoints/verl_opd_haidass_gsm8k/`

### 6.2 🟡 清理冗余容器

- `verl-fresh`：备份容器，可保留或删除
- `verl`（原始）：vLLM 已破坏，建议删除或重建

---

## 七、文件清单

| 文件 | 路径 |
|---|---|
| vLLM 0.18.0 训练脚本 | `/root/run_opd.sh`（容器 verl-vllm 内） |
| 改进版训练脚本 (vLLM 0.10.1) | `/home/verl_latest/examples/on_policy_distillation_trainer/run_haidass_qwen36_distill_gsm8k_npu.sh` |
| 训练日志 | `/home/admin/train_opd_vllm018.log` |
| verl 代码 patch | `/tmp/verl/verl/experimental/teacher_loop/teacher_model.py`（[verl-patch] 标记） |
| 教师模型自定义 config | `/home/models/Qwen3.6-35B-A3B/configuration_qwen3_5_moe.py` |
| 原始教师 config 备份 | `/home/models/Qwen3.6-35B-A3B/config.json.bak` |
| 本汇总文档 | `/home/ycy/pr5837-opd-summary-final.md` |
