#!/bin/bash
# KG-BERT 三元组分类测试脚本（使用已训练好的模型）
# 使用方法:
#   bash scripts/test_kgbert_triple_classifier.sh                        # 默认 CoDEx-S
#   bash scripts/test_kgbert_triple_classifier.sh CoDEx-S                # CoDEx-S
#   bash scripts/test_kgbert_triple_classifier.sh FB15k-237N             # FB15k-237N

# --- 环境配置 ---
export CUDA_VISIBLE_DEVICES=2

# --- 数据集选择 ---
DATASET="${1:-CoDEx-S}"

case "$DATASET" in
    CoDEx-S)
        MAX_SEQ_LENGTH=100
        EVAL_BATCH_SIZE=256
        DO_LOWER_CASE='--do_lower_case'
        ;;
    FB15k-237N)
        MAX_SEQ_LENGTH=200
        EVAL_BATCH_SIZE=512
        DO_LOWER_CASE=''
        ;;
    *)
        echo "错误: 不支持的数据集 '$DATASET'"
        echo "支持的数据集: CoDEx-S, FB15k-237N"
        exit 1
        ;;
esac

# --- 路径配置 ---
DATA_DIR="kg-bert/data/${DATASET}"
MODEL_DIR="kg-bert/output/triple_classifier_${DATASET}"
LOG_DIR='kg-bert/logs'
TIME_STAMP=$(date +%Y%m%d_%H%M%S)

# --- 检查模型 ---
if [ ! -f "${MODEL_DIR}/pytorch_model.bin" ]; then
    echo "错误: 未找到训练好的模型: ${MODEL_DIR}/pytorch_model.bin"
    echo "请先运行训练脚本: bash scripts/train_kgbert_triple_classifier.sh $DATASET"
    exit 1
fi

# --- 检查数据 ---
if [ ! -d "$DATA_DIR" ] || [ ! -f "${DATA_DIR}/test.tsv" ]; then
    echo "错误: 数据路径不存在: $DATA_DIR"
    echo "请先运行: python kg-bert/convert_data.py"
    exit 1
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_triple_classifier_${DATASET}_${TIME_STAMP}.log"

# --- 显示配置信息 ---
echo -e "\n=== KG-BERT 三元组分类测试配置信息 ==="
echo "数据集:          $DATASET"
echo "模型目录:        $MODEL_DIR"
echo "数据路径:        $DATA_DIR"
echo "日志文件:        $LOG_FILE"
echo "---------------------------------"
echo "最大序列长度:     $MAX_SEQ_LENGTH"
echo "评估批次大小:     $EVAL_BATCH_SIZE"
echo "==================================="

# --- 启动测试 ---
python -u kg-bert/run_bert_triple_classifier.py \
    --task_name kg \
    --do_eval \
    --do_predict \
    --data_dir "$DATA_DIR" \
    $DO_LOWER_CASE \
    --bert_model "$MODEL_DIR" \
    --max_seq_length $MAX_SEQ_LENGTH \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --output_dir "$MODEL_DIR" \
    --num_train_epochs 5.0 \
    2>&1 | tee "$LOG_FILE"
