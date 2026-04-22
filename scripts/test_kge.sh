#!/bin/bash
# KGE模型测试脚本 - 加载已训练的checkpoint进行Link Prediction评估
# 使用方法:
#   bash scripts/test_kge.sh                                # FB15k-237N, 自动查找最新checkpoint
#   bash scripts/test_kge.sh CoDEx-S                        # CoDEx-S, 自动查找最新checkpoint
#   bash scripts/test_kge.sh FB15k-237N <config_json_path>  # 指定config文件
#   bash scripts/test_kge.sh FB15k-237N <timestamp>         # 指定时间戳

# --- 环境配置 ---
export NPU_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# --- 解析数据集参数 ---
DATASET="${1:-FB15k-237N}"
case "$DATASET" in
    FB15k-237N|CoDEx-S)
        shift 2>/dev/null || true
        ;;
    *)
        # 不是已知数据集名, 可能是省略数据集直接传了时间戳/config路径
        DATASET="FB15k-237N"
        ;;
esac

# --- 路径配置 (按数据集隔离) ---
OUTPUT_DIR="LLM_Discriminator/train_kge_output/${DATASET}"
LOG_DIR="LLM_Discriminator/logs"

# --- 确定要测试的模型 ---

if [ -n "$1" ]; then
    # 检查参数是配置文件路径还是时间戳
    if [ -f "$1" ]; then
        CONFIG_PATH="$1"
        # 从config路径提取时间戳
        TIME_STAMP=$(basename "$CONFIG_PATH" | sed 's/config_\(.*\)\.json/\1/')
    else
        TIME_STAMP="$1"
        CONFIG_PATH="${OUTPUT_DIR}/config_${TIME_STAMP}.json"
        if [ ! -f "$CONFIG_PATH" ]; then
            echo "错误: 配置文件不存在: $CONFIG_PATH"
            exit 1
        fi
    fi
else
    # 自动查找最新的config文件
    CONFIG_PATH=$(ls -t ${OUTPUT_DIR}/config_*.json 2>/dev/null | head -1)
    if [ -z "$CONFIG_PATH" ]; then
        echo "错误: 在 ${OUTPUT_DIR} 中未找到任何 config_*.json 文件"
        echo "请确认已运行训练脚本: bash scripts/train_kge.sh $DATASET"
        exit 1
    fi
    TIME_STAMP=$(basename "$CONFIG_PATH" | sed 's/config_\(.*\)\.json/\1/')
fi

# 验证checkpoint存在
CKPT_PATH="${OUTPUT_DIR}/checkpoint_${TIME_STAMP}/rotate.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "错误: checkpoint文件不存在: $CKPT_PATH"
    exit 1
fi

LOG_FILE="${LOG_DIR}/test_kge_RotatE_${DATASET}_${TIME_STAMP}.log"
mkdir -p "$LOG_DIR"

# --- 显示配置信息 ---
echo -e "\n=== KGE模型测试配置信息 ==="
echo "数据集:      $DATASET"
echo "Config:      $CONFIG_PATH"
echo "Checkpoint:  $CKPT_PATH"
echo "日志文件:    $LOG_FILE"
echo "============================="

# --- 启动测试 ---
python -u LLM_Discriminator/test_kge.py \
    --config "$CONFIG_PATH" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n测试完成! 结果已保存。"
else
    echo -e "\n测试失败 (退出码: $EXIT_CODE)，请检查日志: $LOG_FILE"
fi

exit $EXIT_CODE
