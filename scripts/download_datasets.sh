#!/bin/bash
# Download MOPD datasets for multi-teacher on-policy distillation
#
# Datasets:
#   1. DeepMath-103K  — 103K math problems (~5GB)
#   2. SciKnowEval    — 28K science QA (~115MB)
#   3. LiveCodeBench  — Competitive programming (~4.5GB)
#   4. ToolAlpaca     — 3.9K tool-use instances (GitHub)
#
# Usage:
#   bash download_datasets.sh              # download all
#   bash download_datasets.sh deepmath     # download one
#   bash download_datasets.sh sciknow livecode toolalpaca  # download multiple
#
# Requirements:
#   pip install huggingface_hub datasets
#   git (for ToolAlpaca)

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-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}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJECT_DIR/.env" ]; then
    source "$PROJECT_DIR/.env"
fi

DATASET_DIR="${DATASET_DIR:-$(pwd)/datasets}"
mkdir -p "$DATASET_DIR"

# Detect CLI: prefer hf, fallback to python huggingface_hub
download_hf() {
    local repo="$1"
    local dest="$2"

    if command -v hf &>/dev/null; then
        hf download "$repo" --repo-type dataset --local-dir "$dest" --resume-download
    else
        echo "  Using Python fallback (huggingface_hub)..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$repo',
    repo_type='dataset',
    local_dir='$dest',
    resume_download=True,
)
print('  Downloaded via Python API')
"
    fi
}

download_deepmath() {
    echo ""
    echo "============================================"
    echo "[1/4] DeepMath-103K"
    echo "  Source: zwhe99/DeepMath-103K"
    echo "  Size:   ~5GB (103,022 math problems)"
    echo "  Domain: Math reasoning (Algebra, Calculus,"
    echo "          Number Theory, Geometry, Probability)"
    echo "  MOPD:   57K samples with difficulty >= 6"
    echo "  HF:     https://huggingface.co/datasets/zwhe99/DeepMath-103K"
    echo "============================================"
    download_hf "zwhe99/DeepMath-103K" "$DATASET_DIR/DeepMath-103K"
    echo "[1/4] DeepMath-103K done ($(du -sh "$DATASET_DIR/DeepMath-103K" 2>/dev/null | cut -f1))"
}

download_sciknow() {
    echo ""
    echo "============================================"
    echo "[2/4] SciKnowEval"
    echo "  Source: hicai-zju/SciKnowEval"
    echo "  Size:   ~115MB (28,392 science QA)"
    echo "  Domain: Biology (8K), Chemistry (9K),"
    echo "          Physics (5K), Materials (6K)"
    echo "  MOPD:   L3 reasoning subset"
    echo "  HF:     https://huggingface.co/datasets/hicai-zju/SciKnowEval"
    echo "============================================"
    download_hf "hicai-zju/SciKnowEval" "$DATASET_DIR/SciKnowEval"
    echo "[2/4] SciKnowEval done ($(du -sh "$DATASET_DIR/SciKnowEval" 2>/dev/null | cut -f1))"
}

download_livecode() {
    echo ""
    echo "============================================"
    echo "[3/4] LiveCodeBench"
    echo "  Source: livecodebench/code_generation_lite"
    echo "  Size:   ~4.5GB (competitive programming)"
    echo "  Domain: AtCoder + LeetCode contest problems"
    echo "  MOPD:   release_v6 (131 eval problems)"
    echo "  HF:     https://huggingface.co/datasets/livecodebench/code_generation_lite"
    echo "============================================"
    download_hf "livecodebench/code_generation_lite" "$DATASET_DIR/LiveCodeBench"
    echo "[3/4] LiveCodeBench done ($(du -sh "$DATASET_DIR/LiveCodeBench" 2>/dev/null | cut -f1))"
}

download_toolalpaca() {
    echo ""
    echo "============================================"
    echo "[4/4] ToolAlpaca"
    echo "  Source: tangqiaoyu/ToolAlpaca (GitHub)"
    echo "  Size:   ~small (3,938 tool-use instances)"
    echo "  Domain: 426 tools, 50 categories"
    echo "          Single-call (2,512) + Multi-call (1,426)"
    echo "  GitHub: https://github.com/tangqiaoyu/ToolAlpaca"
    echo "============================================"
    if [ -d "$DATASET_DIR/ToolAlpaca" ]; then
        echo "  Directory exists, pulling latest..."
        cd "$DATASET_DIR/ToolAlpaca" && git pull
    else
        git clone https://github.com/tangqiaoyu/ToolAlpaca.git "$DATASET_DIR/ToolAlpaca"
    fi
    echo "[4/4] ToolAlpaca done ($(du -sh "$DATASET_DIR/ToolAlpaca" 2>/dev/null | cut -f1))"
}

# Parse arguments
TARGETS=("${@:-all}")

for target in "${TARGETS[@]}"; do
    case "$target" in
        deepmath|1)    download_deepmath ;;
        sciknow|2)     download_sciknow ;;
        livecode|3)    download_livecode ;;
        toolalpaca|4)  download_toolalpaca ;;
        all)
            download_deepmath
            download_sciknow
            download_livecode
            download_toolalpaca
            ;;
        *)
            echo "Unknown dataset: $target"
            echo "Available: deepmath, sciknow, livecode, toolalpaca, all"
            exit 1
            ;;
    esac
done

echo ""
echo "============================================"
echo "Download Summary"
echo "============================================"
echo ""
du -sh "$DATASET_DIR"/* 2>/dev/null || echo "Unable to calculate sizes"
echo ""
echo "Total: $(du -sh "$DATASET_DIR" 2>/dev/null | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Convert to verl parquet format (see scripts/convert_datasets.py)"
echo "  2. Add data_source field to each dataset"
echo "  3. Configure MOPD routing in run_opd.sh"
