#!/bin/bash
# mcts预处理启动脚本
# 使用方法: bash scripts/preprocess_mcts.sh [DATASET]
# for fb15k-237n, threshold=9e-5
# for codex-s, threshold=5e-4

DATASET="${1:-CoDEx-S}"
THRESHOLD=${2:-}

case "$DATASET" in
    CoDEx-S) THRESHOLD="${THRESHOLD:-5e-4}" ;;
    FB15k-237N) THRESHOLD="${THRESHOLD:-9e-5}" ;;
    *)
        echo "不支持的数据集: $DATASET"
        exit 1
        ;;
esac

mkdir -p MCTS/output
python MCTS/preprocess.py \
    --data_folder "data/${DATASET}" \
    --output_file "MCTS/output/processed_data_${DATASET,,}.pth" \
    --threshold "$THRESHOLD"
# mcts预处理启动脚本
# 使用方法: cdko && ackopa && bash scripts/preprocess_mcts.sh
# for fb15k-237n, threshold=9e-5
# for codex-s, threshold=5e-4

python MCTS/preprocess.py \
    --data_folder data/CoDEx-S \
    --output_path MCTS/output/codex-s \
    --threshold 5e-4
