#!/bin/bash
# 微调后的LLM的测试脚本
# 使用方法: bash scripts/test_finetuned_llm.sh

# ==================== 路径设置 ====================
MODEL_PATH='/data/yitingting/models/alpaca-7b'
DATA_PATH='LLM_Discriminator/data/CoDeX-S-test.json'
LORA_PATH='LLM_Discriminator/output/alpaka_7b_codex'
LOG_DIR='LLM_Discriminator/logs'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/test_codex_${TIME_STAMP}.log"

# ==================== GPU 设置 ====================
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=39503
NPROC=$(( $(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c) + 1 ))

# ==================== 环境变量 ====================
export TOKENIZERS_PARALLELISM=false

# 创建目录
mkdir -p $LOG_DIR

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
    LLM_Discriminator/test_with_discriminator.py \
    --base_model $MODEL_PATH \
    --test_data $DATA_PATH \
    --lora_weights $LORA_PATH \
    --root_dir $PWD \
    >> $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
{
    echo "========================================================="
    echo "测试进程已启动, PID: $PID"
    echo "LoRA权重:  $LORA_PATH/"
    echo "日志文件:  $LOG_FILE"
    echo "========================================================="
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: kill $PID"
    echo "========================================================="
} | tee -a "$LOG_FILE"
