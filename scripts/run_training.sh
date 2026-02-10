#!/bin/bash
# ============================================================
# COMET Reference-Free 학습 실행 스크립트
# ============================================================
# 사용법:
#   bash scripts/run_training.sh [approach] [options]
#
# approach:
#   mini     - 미니 테스트 (파이프라인 확인용)
#   scratch1 - 접근법 1: ReferencelessRegression from scratch
#   scratch2 - 접근법 2: UnifiedMetric QE from scratch
#   finetune - 접근법 3: COMETKiwi fine-tuning (추천)
#   ft-qe    - 접근법 4: ReferencelessRegression fine-tuning
#
# options:
#   --seed N           : 랜덤 시드 (기본값: 12)
#   --checkpoint PATH  : Fine-tuning용 체크포인트 경로
#   --gpus N           : GPU 수 (기본값: 1)
#
# 예시:
#   bash scripts/run_training.sh mini
#   bash scripts/run_training.sh scratch1 --seed 42
#   bash scripts/run_training.sh finetune --checkpoint /path/to/model.ckpt
# ============================================================

set -e

APPROACH=${1:-"mini"}
shift || true

# 기본값
SEED=12
CHECKPOINT=""
GPUS=1

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

echo "============================================================"
echo "COMET Reference-Free 학습"
echo "============================================================"
echo "  접근법:     $APPROACH"
echo "  랜덤 시드:  $SEED"
echo "  GPU 수:     $GPUS"
echo "  체크포인트:  ${CHECKPOINT:-'없음 (from scratch)'}"
echo "============================================================"

# 설정 파일 선택
case $APPROACH in
    mini)
        CONFIG="configs/models/en-ko-qe/approach_mini_test.yaml"
        echo "[INFO] 미니 테스트 실행 (파이프라인 확인)"
        ;;
    scratch1)
        CONFIG="configs/models/en-ko-qe/approach1_referenceless_scratch.yaml"
        echo "[INFO] 접근법 1: ReferencelessRegression from scratch"
        ;;
    scratch2)
        CONFIG="configs/models/en-ko-qe/approach2_unified_qe_scratch.yaml"
        echo "[INFO] 접근법 2: UnifiedMetric QE from scratch"
        ;;
    finetune)
        CONFIG="configs/models/en-ko-qe/approach3_finetune_cometkiwi.yaml"
        echo "[INFO] 접근법 3: COMETKiwi fine-tuning"
        if [ -z "$CHECKPOINT" ]; then
            echo "[ERROR] Fine-tuning에는 --checkpoint 옵션이 필요합니다."
            echo "  먼저 체크포인트를 다운로드하세요:"
            echo "  python scripts/download_checkpoint.py --model Unbabel/wmt22-cometkiwi-da"
            exit 1
        fi
        ;;
    ft-qe)
        CONFIG="configs/models/en-ko-qe/approach4_referenceless_finetune_qe.yaml"
        echo "[INFO] 접근법 4: ReferencelessRegression fine-tuning"
        if [ -z "$CHECKPOINT" ]; then
            echo "[ERROR] Fine-tuning에는 --checkpoint 옵션이 필요합니다."
            exit 1
        fi
        ;;
    *)
        echo "[ERROR] Unknown approach: $APPROACH"
        echo "  Available: mini, scratch1, scratch2, finetune, ft-qe"
        exit 1
        ;;
esac

# 설정 파일 확인
if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config file not found: $CONFIG"
    exit 1
fi

# 데이터 파일 확인
echo ""
echo "[CHECK] Verifying data files..."
if [ "$APPROACH" == "mini" ]; then
    TRAIN_FILE="data/en-ko-qe/mini_train.csv"
    VAL_FILE="data/en-ko-qe/mini_val.csv"
elif [ "$APPROACH" == "scratch1" ] || [ "$APPROACH" == "ft-qe" ]; then
    TRAIN_FILE="data/en-ko-qe/referenceless_train.csv"
    VAL_FILE="data/en-ko-qe/referenceless_val.csv"
else
    TRAIN_FILE="data/en-ko-qe/unified_qe_train.csv"
    VAL_FILE="data/en-ko-qe/unified_qe_val.csv"
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "[ERROR] Training data not found: $TRAIN_FILE"
    echo "  먼저 데이터 전처리를 실행하세요:"
    echo "  python scripts/prepare_data.py --input_dir /path/to/train_data --output_dir data/en-ko-qe"
    exit 1
fi
if [ ! -f "$VAL_FILE" ]; then
    echo "[ERROR] Validation data not found: $VAL_FILE"
    exit 1
fi

TRAIN_LINES=$(wc -l < "$TRAIN_FILE")
VAL_LINES=$(wc -l < "$VAL_FILE")
echo "  Train: $TRAIN_FILE ($((TRAIN_LINES - 1)) samples)"
echo "  Val:   $VAL_FILE ($((VAL_LINES - 1)) samples)"

# GPU 확인
echo ""
echo "[CHECK] GPU status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "  nvidia-smi not available"
fi

# 학습 명령 구성
CMD="comet-train --cfg $CONFIG --seed_everything $SEED"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --load_from_checkpoint $CHECKPOINT"
fi

echo ""
echo "[RUN] $CMD"
echo "============================================================"
echo ""

# 학습 실행
eval $CMD

echo ""
echo "============================================================"
echo "[DONE] Training completed!"
echo "  체크포인트 위치: lightning_logs/"
echo "  평가 실행: python scripts/evaluate_model.py --checkpoint <path>"
echo "============================================================"
