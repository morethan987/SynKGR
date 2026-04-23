#!/bin/bash
# KGE模型训练启动脚本
# 使用方法:
#   bash scripts/train_kge.sh                        # 默认使用 FB15k-237N
#   bash scripts/train_kge.sh FB15k-237N             # 显式指定 FB15k-237N
#   bash scripts/train_kge.sh CoDEx-S                # 使用 CoDEx-S 数据集

# NPU环境配置
export NPU_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# --- 数据集选择 ---
DATASET="${1:-FB15k-237N}"

case "$DATASET" in
    FB15k-237N)
        RAW_DATA_PATH='data/FB15k-237N'
        ;;
    CoDEx-S)
        RAW_DATA_PATH='data/CoDEx-S'
        ;;
    *)
        echo "错误: 不支持的数据集 '$DATASET'"
        echo "支持的数据集: FB15k-237N, CoDEx-S"
        exit 1
        ;;
esac

# --- 路径配置 ---
DATA_PATH="openke/${DATASET}"
OUTPUT_DIR="openke/output/${DATASET}"
LOG_DIR="openke/logs"

# --- 数据预处理 ---
# 检查是否需要将原始数据转换为 OpenKE 格式
if [ ! -d "$DATA_PATH" ] || [ ! -f "${DATA_PATH}/train2id.txt" ]; then
    echo "OpenKE格式数据不存在，开始预处理..."
    if [ ! -d "$RAW_DATA_PATH" ]; then
        echo "错误: 原始数据路径不存在: $RAW_DATA_PATH"
        exit 1
    fi
    python openke/preprocess_for_openke.py --src_dir "$RAW_DATA_PATH" --dst_dir "$DATA_PATH"
    if [ $? -ne 0 ]; then
        echo "错误: 数据预处理失败"
        exit 1
    fi
    echo "数据预处理完成。"
fi

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_kge_RotatE_${DATASET}_${TIME_STAMP}.log"

# --- 超参数配置 ---
MODEL_NAME="RotatE"
DIMENSION=512         # embedding dimension d_e = d_r = 512
NEGATIVE_SAMPLES=32   # sample K = 32 negative samples

MARGIN=6.0            # margin γ is tuned among {0, 4, 6, 8, 12}. 6.0是一个很好的起始值.

BATCH_SIZE=2048       # KGE训练通常使用较大的batch size以提升效率和稳定性
LEARNING_RATE=2e-5    # 一个在KGE任务中常见且有效的学习率
OPTIMIZER="Adam"      # Adam是KGE训练中最常用的优化器
EPOCHS=3000           # 训练轮数, 可根据收敛情况调整

# --- 脚本执行 ---

# 检查路径是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据路径不存在: $DATA_PATH"
    echo "请先运行: python openke/preprocess_for_openke.py"
    exit 1
fi

if [ ! -f "${DATA_PATH}/train2id.txt" ]; then
    echo "错误: 未找到 OpenKE 格式数据 (train2id.txt)"
    echo "请先运行: python openke/preprocess_for_openke.py"
    exit 1
fi

# 创建日志和输出目录
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# 显示配置信息
echo -e "\n=== KGE模型训练配置信息 ==="
echo "数据集:      $DATASET"
echo "模型:        $MODEL_NAME"
echo "数据路径:    $DATA_PATH"
echo "输出目录:    $OUTPUT_DIR"
echo "日志文件:    $LOG_FILE"
echo "---------------------------------"
echo "维度 (Dimension): $DIMENSION"
echo "批次大小 (Batch Size): $BATCH_SIZE"
echo "学习率 (Learning Rate): $LEARNING_RATE"
echo "边际值 (Margin γ): $MARGIN"
echo "优化器 (Optimizer): $OPTIMIZER"
echo "负采样数 (Negative Samples K): $NEGATIVE_SAMPLES"
echo "训练轮数 (Epochs): $EPOCHS"
echo "==================================="

# 用户确认
read -p "是否继续训练? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "训练已取消。"
    exit 0
fi

# 启动训练脚本
# -u 参数实时刷新日志缓冲区
nohup python -u LLM_Discriminator/train_kge.py \
    --model "$MODEL_NAME" \
    --in_path "$DATA_PATH" \
    --out_path "$OUTPUT_DIR" \
    --time_stamp "$TIME_STAMP" \
    --dimension "$DIMENSION" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --margin "$MARGIN" \
    --optimizer "$OPTIMIZER" \
    --neg_ent "$NEGATIVE_SAMPLES" \
    --epochs "$EPOCHS" \
    --test_triple \
    >> "$LOG_FILE" 2>&1 &

# 获取进程ID并提示用户
PID=$!
{
    echo "========================================================="
    echo "KGE训练进程已启动!"
    echo "数据集: $DATASET"
    echo "进程ID (PID): $PID"
    echo "日志文件: $LOG_FILE"
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止进程: kill $PID"
    echo "========================================================="
} | tee -a "$LOG_FILE"
