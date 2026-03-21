"""
CATM: Class-specific Adaptive Thresholding with Momentum
Inspired by ST-SGG (ICLR 2024): Adaptive Self-training Framework for Fine-grained SGG

This module discovers missing relation annotations in the VG dataset by:
1. Using the model's own predictions to assign pseudo-labels to background (unannotated) relation pairs
2. Maintaining per-class adaptive thresholds via exponential moving average (EMA)
3. Weighting pseudo-label loss relative to ground-truth loss

Key insight: ~93% of object pairs in VG have no relation label (background), but many
actually have valid relations. CATM recovers these missing annotations during training.
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class CATMPseudoLabeler:
    """
    Class-specific Adaptive Thresholding with Momentum.
    
    Maintains per-class confidence thresholds that adapt during training:
    - If many samples exceed threshold → increase threshold (be more selective)
    - If few samples exceed threshold → decrease threshold (be more permissive)
    """
    
    def __init__(self, num_classes=51, init_threshold=0.4, 
                 momentum_inc=0.999, momentum_dec=0.99,
                 pseudo_weight=0.5, warmup_iter=2000):
        self.num_classes = num_classes
        self.pseudo_weight = pseudo_weight
        self.warmup_iter = warmup_iter
        
        # Per-class adaptive thresholds (class 0 = background, never pseudo-label it)
        self.thresholds = torch.ones(num_classes).cuda() * init_threshold
        self.thresholds[0] = 1.0  # never assign pseudo-label to background class
        
        self.momentum_inc = momentum_inc
        self.momentum_dec = momentum_dec
        
        # Statistics tracking
        self.total_pseudo = 0
        self.total_bg = 0
        self.pseudo_per_class = torch.zeros(num_classes)
        
    def compute_pseudo_loss(self, relation_logits, rel_labels, iteration, 
                            cls_weight=None):
        """
        Compute pseudo-label loss for background (unannotated) relation pairs.
        
        Args:
            relation_logits: list of [num_rels_i, num_classes] tensors
            rel_labels: list of [num_rels_i] label tensors  
            iteration: current training iteration
            cls_weight: optional CB-Loss class weights tensor
            
        Returns:
            pseudo_loss: scalar loss tensor (0 if no pseudo-labels assigned)
            num_pseudo: number of pseudo-labeled pairs this batch
        """
        if iteration < self.warmup_iter:
            return torch.tensor(0.0).cuda(), 0
        
        # Concatenate all batches
        all_logits = torch.cat(relation_logits, dim=0)  # [N, num_classes]
        all_labels = torch.cat(rel_labels, dim=0)        # [N]
        
        # Find background (unannotated) pairs
        bg_mask = (all_labels == 0)
        if not bg_mask.any():
            return torch.tensor(0.0).cuda(), 0
        
        bg_logits = all_logits[bg_mask]  # [M, num_classes]
        
        # Get model confidence for each foreground class
        bg_probs = F.softmax(bg_logits, dim=1)
        # Exclude background class (index 0) from pseudo-label candidates
        fg_probs = bg_probs[:, 1:]  # [M, num_classes-1]
        
        max_prob, max_idx = fg_probs.max(dim=1)  # [M], [M]
        max_class = max_idx + 1  # offset by 1 (background is class 0)
        
        # Apply per-class thresholds
        class_thresholds = self.thresholds[max_class]  # [M]
        pseudo_mask = max_prob > class_thresholds
        
        if not pseudo_mask.any():
            return torch.tensor(0.0).cuda(), 0
        
        # Assign pseudo-labels
        pseudo_logits = bg_logits[pseudo_mask]
        pseudo_labels = max_class[pseudo_mask]
        num_pseudo = pseudo_mask.sum().item()
        
        # Compute weighted cross-entropy loss on pseudo-labeled pairs
        if cls_weight is not None:
            pseudo_loss = F.cross_entropy(pseudo_logits, pseudo_labels, cls_weight)
        else:
            pseudo_loss = F.cross_entropy(pseudo_logits, pseudo_labels)
        
        pseudo_loss = pseudo_loss * self.pseudo_weight
        
        # Update per-class thresholds via EMA
        self._update_thresholds(max_class, max_prob, pseudo_mask)
        
        # Update statistics
        self.total_pseudo += num_pseudo
        self.total_bg += bg_mask.sum().item()
        
        # Logging every 1000 iterations
        if iteration % 1000 == 0:
            ratio = num_pseudo / max(bg_mask.sum().item(), 1) * 100
            logger.info(
                f"[CATM] iter={iteration}: {num_pseudo}/{bg_mask.sum().item()} "
                f"bg pairs pseudo-labeled ({ratio:.1f}%), "
                f"avg_threshold={self.thresholds[1:].mean():.4f}, "
                f"pseudo_loss={pseudo_loss.item():.4f}"
            )
        
        return pseudo_loss, num_pseudo
    
    def _update_thresholds(self, predicted_classes, confidences, accepted_mask):
        """Update per-class thresholds based on prediction statistics."""
        with torch.no_grad():
            for c in range(1, self.num_classes):
                class_mask = (predicted_classes == c)
                if not class_mask.any():
                    continue
                
                # Check acceptance rate for this class
                class_accepted = accepted_mask[class_mask].float().mean()
                
                if class_accepted > 0.5:
                    # Too many accepted → increase threshold (be more selective)
                    self.thresholds[c] = (
                        self.momentum_inc * self.thresholds[c] + 
                        (1 - self.momentum_inc) * confidences[class_mask].mean()
                    )
                else:
                    # Too few accepted → decrease threshold (be more permissive)
                    self.thresholds[c] = (
                        self.momentum_dec * self.thresholds[c] + 
                        (1 - self.momentum_dec) * confidences[class_mask].mean()
                    )
                
                # Clamp thresholds to reasonable range
                self.thresholds[c] = self.thresholds[c].clamp(0.1, 0.9)
