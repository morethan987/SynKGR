#!/bin/bash
# 微调启动脚本
# 使用方法: bash scripts/run_finetune_codex.sh

# ==================== 路径设置 ====================
MODEL_PATH='/data/yitingting/models/alpaca-7b'
DATA_PATH='LLM_Discriminator/data/CoDeX-S-test.json'
KGE_MODEL='LLM_Discriminator/data/CoDeX-S-rotate.pth'
OUTPUT_DIR='LLM_Discriminator/output/alpaka_7b_codex'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)

# ==================== GPU 设置 ====================
export CUDA_VISIBLE_DEVICES=0,2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503
NPROC=$(( $(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c) + 1 ))

# ==================== 环境变量 ====================
export TOKENIZERS_PARALLELISM=false

# 创建输出目录
mkdir -p $OUTPUT_DIR
LOG_FILE="$OUTPUT_DIR/train_${TIME_STAMP}.log"

# 显示GPU信息
echo "GPU信息:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi 命令不可用"
fi
# 让用户确认是否继续
read -p "Continue? (Y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" && "$CONFIRM" != "" ]]; then
    echo "Canceled!"
    exit 0
fi

nohup torchrun \
    --nnodes=1 \
    --nproc_per_node=$NPROC \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    LLM_Discriminator/finetune.py \
    --base_model $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_epochs 3 \
    --lora_r 64 \
    --learning_rate 3e-4 \
    --batch_size 8 \
    --micro_batch_size 8 \
    --num_prefix 1 \
    --kge_model $KGE_MODEL \
    --lora_target_modules '["q_proj","k_proj","v_proj","o_proj"]' \
    --llm_dim 4096 \
    --train_sampling_strategy "group_by_length" \
    >> $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
{
    echo "========================================================="
    echo "LoRA微调进程已启动, PID: $PID"
    echo "输出目录: $OUTPUT_DIR/"
    echo "  - 模型权重: $OUTPUT_DIR/adapter_model.bin"
    echo "  - KG嵌入:   $OUTPUT_DIR/embeddings.pth"
    echo "  - 训练日志: $LOG_FILE"
    echo "  - 检查点:   $OUTPUT_DIR/checkpoint-*/"
    echo "========================================================="
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: kill $PID"
    echo "========================================================="
} | tee -a "$LOG_FILE"
