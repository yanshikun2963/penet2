# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .sampling import make_roi_relation_samp_processor

class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class with optional CATM pseudo-labeling.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)

        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

        # CATM pseudo-labeling
        self.catm = None
        if hasattr(cfg.MODEL, 'CATM_ENABLE') and cfg.MODEL.CATM_ENABLE:
            from .catm import CATMPseudoLabeler
            self.catm = CATMPseudoLabeler(
                num_classes=cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES,
                init_threshold=cfg.MODEL.CATM_THRESHOLD,
                momentum_inc=cfg.MODEL.CATM_MOMENTUM_INC,
                momentum_dec=cfg.MODEL.CATM_MOMENTUM_DEC,
                pseudo_weight=cfg.MODEL.CATM_PSEUDO_WEIGHT,
                warmup_iter=cfg.MODEL.CATM_WARMUP_ITER,
            )
        self._iteration = 0

    def forward(self, features, proposals, targets=None, logger=None):
        if self.training:
            self._iteration += 1
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets)
        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals)

        roi_features = self.box_feature_extractor(features, proposals)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)

        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        else:
            union_features = None
        
        refine_logits, relation_logits, add_losses, add_data = self.predictor(
            proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)

        # for test
        if not self.training:
            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
            return roi_features, result, {}

        loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits)

        if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine[0], loss_refine_att=loss_refine[1])
        elif hasattr(self.cfg.MODEL, 'reweight_fineloss') and self.cfg.MODEL.reweight_fineloss:
            output_losses = dict(loss_refine_obj=loss_refine)
        else:
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)

        output_losses.update(add_losses)

        # CATM pseudo-labeling: assign pseudo-labels to background pairs
        if self.catm is not None and self.training:
            cls_weight = getattr(self.predictor, 'final_cls_weight', None)
            pseudo_loss, num_pseudo = self.catm.compute_pseudo_loss(
                relation_logits, rel_labels, self._iteration, cls_weight)
            if num_pseudo > 0:
                output_losses['loss_pseudo'] = pseudo_loss

        return roi_features, proposals, output_losses


def build_roi_relation_head(cfg, in_channels):
    return ROIRelationHead(cfg, in_channels)
