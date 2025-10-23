#!/bin/bash
# 不同关系类型的性能测试启动脚本
# 使用方法: cdko && ackopa && bash scripts/test_relation_type.sh

# 路径设置
DATA_SET='FB15K-237N'
# DATA_SET='CoDEx-S'
OUTPUT_DIR='loss_restraint_KGE_model/output/fb15k-237n'
# OUTPUT_DIR='loss_restraint_KGE_model/output/codex-s'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)

# 创建目录及文件
mkdir -p $OUTPUT_DIR

# 显示NPU信息
echo "NPU信息:"
if command -v npu-smi &> /dev/null; then
    npu-smi info
else
    nvidia-smi
fi
# 让用户确认是否继续
read -p "Continue? (Y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" && "$CONFIRM" != "" ]]; then
    echo "Canceled!"
    exit 0
fi

python loss_restraint_KGE_model/run.py \
    --restore \
    --name test_run_20251012_191309 \
    --mode test_relation_type \
    --time_string $TIME_STAMP \
    --data $DATA_SET \
    --save $OUTPUT_DIR \
    --score_func conve \
    --opn corr \
    --adapt_aggr 1 \
    --loss_delta 0.002 \
    --batch 256 \
    --lr 5e-4 \
    --epoch 500 \
    --gpu 2
