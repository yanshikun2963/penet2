#!/bin/bash
# =============================================================
# Exp-B: PE-NET + CB-Loss + ST-SGG CATM Pseudo-Labeling
# CATM Threshold: 0.3 | GPU: 1x RTX 5090
# =============================================================

set -e

# ===== 你需要修改的路径 =====
STRONG_BASE_CKPT="./checkpoints/PE-NET_Reweight_PredCls_beta9999/model_0032000.pth"
DETECTOR_CKPT="./checkpoints/pretrained_faster_rcnn/model_final.pth"
GLOVE="./datasets/vg/"

# ===== 实验参数 =====
MAX_ITER=30000
LR=0.0001
STEPS="(18000,24000)"
OUTPUT_DIR="./checkpoints/finetune_stsgg_low"
MODEL_NAME="Exp-B-STSGG-thresh0.3"

# ===== 环境变量 =====
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export PYTHONPATH="$(pwd):$PYTHONPATH"

# ===== 检查 =====
echo "============================================="
echo "  ${MODEL_NAME}"
echo "============================================="
echo "[CHECK] 强基座 checkpoint..."
if [ ! -f "$STRONG_BASE_CKPT" ]; then
    echo "[ERROR] 找不到强基座checkpoint: $STRONG_BASE_CKPT"
    exit 1
fi
echo "  OK: $(ls -lh $STRONG_BASE_CKPT | awk '{print $5, $NF}')"

echo "[CHECK] VG数据集..."
if [ ! -f "./datasets/vg/VG-SGG-with-attri.h5" ]; then
    echo "[ERROR] 找不到 datasets/vg/VG-SGG-with-attri.h5"
    exit 1
fi
echo "  OK"

mkdir -p ${OUTPUT_DIR}

echo ""
echo "[START] CATM threshold=0.3, 训练开始: $(date)"
echo "============================================="
echo ""

stdbuf -oL python3 tools/relation_train_net.py \
  --config-file "configs/finetune_stsgg.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
  MODEL.reweight_fineloss True \
  MODEL.num_beta 0.9999 \
  MODEL.CATM_ENABLE True \
  MODEL.CATM_THRESHOLD 0.3 \
  MODEL.CATM_PSEUDO_WEIGHT 0.5 \
  MODEL.CATM_WARMUP_ITER 2000 \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 \
  TEST.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER ${MAX_ITER} \
  SOLVER.BASE_LR ${LR} \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  SOLVER.STEPS "${STEPS}" \
  SOLVER.VAL_PERIOD 5000 \
  SOLVER.CHECKPOINT_PERIOD 5000 \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  GLOVE_DIR ${GLOVE} \
  MODEL.PRETRAINED_DETECTOR_CKPT ${STRONG_BASE_CKPT} \
  OUTPUT_DIR ${OUTPUT_DIR} \
  2>&1 | tee ${OUTPUT_DIR}/train.log

echo ""
echo "[DONE] 训练完成: $(date)"
