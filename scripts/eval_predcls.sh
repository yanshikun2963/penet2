#!/bin/bash
# =============================================================
# 评估脚本：测试fine-tune后的模型在PredCls任务上的性能
# Usage: bash scripts/eval_predcls.sh <checkpoint_path>
# Example: bash scripts/eval_predcls.sh ./checkpoints/finetune_ietrans_10k/model_0010000.pth
# =============================================================

CKPT=${1:?"请提供checkpoint路径，例如: bash scripts/eval_predcls.sh ./checkpoints/finetune_ietrans_10k/model_0010000.pth"}
OUTPUT_DIR="$(dirname $CKPT)/eval_predcls"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "============================================="
echo "  评估 PredCls: $CKPT"
echo "============================================="

mkdir -p ${OUTPUT_DIR}

stdbuf -oL python3 tools/relation_test_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
  DTYPE "float32" \
  TEST.IMS_PER_BATCH 1 \
  GLOVE_DIR /root/autodl-tmp/penet-main/Datasets/VG/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ${CKPT} \
  OUTPUT_DIR ${OUTPUT_DIR} \
  MODEL.WEIGHT ${CKPT} \
  2>&1 | tee ${OUTPUT_DIR}/eval.log

echo ""
echo "[DONE] 评估完成，结果在: ${OUTPUT_DIR}/eval.log"
