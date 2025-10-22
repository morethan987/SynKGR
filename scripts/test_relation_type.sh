#!/bin/bash
# 不同关系类型的性能测试启动脚本
# 使用方法: cdko && ackopa && bash scripts/test_relation_type.sh

# DATASET="FB15K-237N"
DATASET="CoDEx-S"

# MODEL_PATH="loss_restraint_KGE_model/output/fb15k-237n/test_run_20251012_191309.pth"
MODEL_PATH="loss_restraint_KGE_model/output/codex-s/repeate_test_run_20251023_015036.pth"

# SAVE_DIR='loss_restraint_KGE_model/output/fb15k-237n'
SAVE_DIR='loss_restraint_KGE_model/output/codex-s'


python tests/relation_type.py \
    --dataset $DATASET \
    --model compgcn \
    --score_func conve \
    --model_path $MODEL_PATH \
    --save_dir $SAVE_DIR \
    --root_dir $PWD \
    --adapt_aggr 1 \
    --gpu 2
