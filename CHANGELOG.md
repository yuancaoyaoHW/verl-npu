# Changelog

## [0.2.0] - 2026-06-04

### 训练成功启动

- 修复 3 个启动 bug，训练已在后台运行 (7005 steps, ~194h)
- 创建 git 项目记录所有改动

### Bug Fixes

1. **qwen3_5_moe CONFIG_MAPPING 注册** (`patches/teacher_model.py`)
   - transformers 4.57.6 不认识 `qwen3_5_moe` 架构
   - `trust_remote_code=True` + `auto_map` 方案在 Ray 多进程下不可行 (pickle 序列化失败)
   - 改为在 verl 代码中用 `importlib` 加载自定义 config 类并注册到 `CONFIG_MAPPING`

2. **移除 PYTORCH_NPU_ALLOC_CONF** (`scripts/run_opd.sh`)
   - `expandable_segments:True` 与 vllm-ascend 0.18.0 的 `CaMemAllocator` 冲突
   - PR #5837 脚本是给 vLLM 0.10.1 写的，0.18.0 不需要此设置

3. **锁定 transformers==4.57.6** (`requirements-pinned.txt`)
   - 排查过程中意外升级到 5.10.1，导致 vLLM spawn 子进程崩溃
   - 5.x 重组了内部模块 (`image_transforms`, `modeling_auto` 等被移除/移动)
   - 同时需降级 `huggingface-hub` 到 `<1.0`

### Added

- `CHANGELOG.md` — 变更记录
- `requirements-pinned.txt` — 关键依赖版本锁定
- `docker/docker-run.sh` — 容器启动脚本
- `patches/apply-patch.sh` — patch 应用脚本

## [0.1.0] - 2026-06-04

### 初始环境搭建

- 创建 `verl-vllm` 容器 (quay.io/ascend/vllm-ascend:v0.18.0)
- 解决 NPU 可见性问题 (5 轮调试)
- 修复 Haidass 模型缺少 tokenizer 文件
- 创建训练脚本 `run_opd.sh` (适配 vLLM 0.18.0)
- 创建参考脚本 `run_haidass_qwen36_distill_gsm8k_npu.sh`
