#!/bin/bash
# KG-BERT 三元组分类训练脚本
# 使用方法:
#   bash scripts/train_kgbert_triple_classifier.sh                     # 默认 CoDEx-S
#   bash scripts/train_kgbert_triple_classifier.sh CoDEx-S             # CoDEx-S
#   bash scripts/train_kgbert_triple_classifier.sh FB15k-237N          # FB15k-237N

# --- 环境配置 ---
export CUDA_VISIBLE_DEVICES=2

# --- 数据集选择 ---
DATASET="${1:-CoDEx-S}"

case "$DATASET" in
    CoDEx-S)
        BERT_MODEL='bert-base-uncased'
        MAX_SEQ_LENGTH=100
        TRAIN_BATCH_SIZE=32
        EVAL_BATCH_SIZE=256
        LEARNING_RATE=5e-5
        NUM_EPOCHS=5.0
        WARMUP_PROPORTION=0.1
        GRADIENT_ACCUMULATION_STEPS=1
        ;;
    FB15k-237N)
        BERT_MODEL='bert-base-cased'
        MAX_SEQ_LENGTH=200
        TRAIN_BATCH_SIZE=32
        EVAL_BATCH_SIZE=512
        LEARNING_RATE=5e-5
        NUM_EPOCHS=3.0
        WARMUP_PROPORTION=0.1
        GRADIENT_ACCUMULATION_STEPS=1
        ;;
    *)
        echo "错误: 不支持的数据集 '$DATASET'"
        echo "支持的数据集: CoDEx-S, FB15k-237N"
        exit 1
        ;;
esac

# --- 路径配置 ---
DATA_DIR="kg-bert/data/${DATASET}"
OUTPUT_DIR="kg-bert/output/triple_classifier_${DATASET}"
LOG_DIR='kg-bert/logs'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)

# --- 检查数据 ---
if [ ! -d "$DATA_DIR" ] || [ ! -f "${DATA_DIR}/train.tsv" ]; then
    echo "错误: 数据路径不存在: $DATA_DIR"
    echo "请先运行: python kg-bert/convert_data.py"
    exit 1
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/triple_classifier_${DATASET}_${TIME_STAMP}.log"

# --- 清理旧输出目录 ---
if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# --- 显示配置信息 ---
echo -e "\n=== KG-BERT 三元组分类训练配置信息 ==="
echo "数据集:          $DATASET"
echo "BERT模型:        $BERT_MODEL"
echo "数据路径:        $DATA_DIR"
echo "输出目录:        $OUTPUT_DIR"
echo "日志文件:        $LOG_FILE"
echo "---------------------------------"
echo "最大序列长度:     $MAX_SEQ_LENGTH"
echo "训练批次大小:     $TRAIN_BATCH_SIZE"
echo "评估批次大小:     $EVAL_BATCH_SIZE"
echo "学习率:          $LEARNING_RATE"
echo "训练轮数:        $NUM_EPOCHS"
echo "梯度累积步数:     $GRADIENT_ACCUMULATION_STEPS"
echo "==================================="

# --- 用户确认 ---
read -p "是否继续训练? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "训练已取消。"
    exit 0
fi

# --- 启动训练 ---
nohup python -u kg-bert/run_bert_triple_classifier.py \
    --task_name kg \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir "$DATA_DIR" \
    --bert_model "$BERT_MODEL" \
    --max_seq_length $MAX_SEQ_LENGTH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --warmup_proportion $WARMUP_PROPORTION \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir "$OUTPUT_DIR" \
    >> "$LOG_FILE" 2>&1 &

PID=$!
{
    echo "========================================================="
    echo "KG-BERT 三元组分类训练进程已启动!"
    echo "数据集: $DATASET"
    echo "进程ID (PID): $PID"
    echo "日志文件: $LOG_FILE"
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: kill $PID"
    echo "========================================================="
} | tee -a "$LOG_FILE"
