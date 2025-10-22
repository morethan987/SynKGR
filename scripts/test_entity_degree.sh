#!/bin/bash
# 不同度数的实体的性能测试启动脚本
# 使用方法: cdko && ackopa && bash scripts/test_entity_degree.sh


python tester.py \
    --root_dir $PWD \
    --model_path checkpoints/model.pth \
    --save_dir checkpoints/ \
    --dataset FB15k-237 \
    --model compgcn \
    --score_func conve \
    --gpu 0
