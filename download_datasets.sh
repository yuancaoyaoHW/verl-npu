#!/bin/bash
# Download MOPD datasets
set -e

DATASET_DIR="/home/ycy/haidass-opd-npu/datasets"
mkdir -p "$DATASET_DIR"

echo "=========================================="
echo "Starting MOPD dataset downloads"
echo "Target directory: $DATASET_DIR"
echo "Time: $(date)"
echo "=========================================="

# 1. DeepMath-103K (HuggingFace)
echo ""
echo "[1/4] Downloading DeepMath-103K..."
echo "Source: zwhe99/DeepMath-103K"
echo "Size: ~5GB (103K math problems)"
huggingface-cli download zwhe99/DeepMath-103K \
  --repo-type dataset \
  --local-dir "$DATASET_DIR/DeepMath-103K" \
  --resume-download
echo "[1/4] DeepMath-103K completed"

# 2. SciKnowEval (HuggingFace)
echo ""
echo "[2/4] Downloading SciKnowEval..."
echo "Source: hicai-zju/SciKnowEval"
echo "Size: ~115MB (28K science QA)"
huggingface-cli download hicai-zju/SciKnowEval \
  --repo-type dataset \
  --local-dir "$DATASET_DIR/SciKnowEval" \
  --resume-download
echo "[2/4] SciKnowEval completed"

# 3. LiveCodeBench (HuggingFace)
echo ""
echo "[3/4] Downloading LiveCodeBench..."
echo "Source: livecodebench/code_generation_lite"
echo "Size: ~4.5GB (competitive programming)"
huggingface-cli download livecodebench/code_generation_lite \
  --repo-type dataset \
  --local-dir "$DATASET_DIR/LiveCodeBench" \
  --resume-download
echo "[3/4] LiveCodeBench completed"

# 4. ToolAlpaca (GitHub)
echo ""
echo "[4/4] Downloading ToolAlpaca..."
echo "Source: tangqiaoyu/ToolAlpaca"
echo "Size: ~small (3.9K tool-use instances)"
if [ -d "$DATASET_DIR/ToolAlpaca" ]; then
  echo "ToolAlpaca directory exists, pulling latest..."
  cd "$DATASET_DIR/ToolAlpaca" && git pull
else
  git clone https://github.com/tangqiaoyu/ToolAlpaca.git "$DATASET_DIR/ToolAlpaca"
fi
echo "[4/4] ToolAlpaca completed"

echo ""
echo "=========================================="
echo "All downloads completed!"
echo "Time: $(date)"
echo "=========================================="
echo ""
echo "Dataset summary:"
du -sh "$DATASET_DIR"/* 2>/dev/null || echo "Unable to calculate sizes"
echo ""
echo "Directory structure:"
ls -la "$DATASET_DIR"
