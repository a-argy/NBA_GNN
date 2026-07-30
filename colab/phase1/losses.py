"""
Phase 1 loss functions for GNN shot prediction.

All functions accept raw logits (NOT log-probabilities).

Available loss types (via get_loss_fn factory):
  'ce'           – standard cross-entropy (equivalent to baseline NLL + log_softmax)
  'label_smooth' – cross-entropy with label smoothing (smoothing ∈ [0,1])
  'focal'        – focal loss FL = -(1-p_t)^γ · log(p_t)  (gamma ≥ 0)

Rationale
---------
Standard CE treats all shots equally.  Label smoothing prevents overconfidence
on a noisy signal (release mechanics after the snapshot are not observed).
Focal loss concentrates gradient on ambiguous shots, down-weighting easy
examples the model already classifies correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Reduces to standard cross-entropy when gamma=0.

    Args:
        gamma        : focusing parameter (0 = CE, larger → more focus on hard)
        class_weights: optional 1-D tensor of per-class weights (same as CE weight)
    """

    def __init__(self, gamma=1.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        Args:
            logits  : [N, C] raw logits
            targets : [N]   long integer class labels

        Returns:
            scalar mean focal loss
        """
        # Per-sample CE loss (unreduced)
        ce = F.cross_entropy(
            logits, targets, weight=self.class_weights, reduction='none'
        )
        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce)
        # Focal weight: down-weight easy examples (high p_t → small weight)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def get_loss_fn(loss_type, **kwargs):
    """
    Factory that returns a callable loss function accepting (logits, targets).

    Args:
        loss_type : 'ce' | 'label_smooth' | 'focal'
        **kwargs  :
            smoothing (float) – label smoothing coefficient for 'label_smooth'
            gamma     (float) – focusing parameter for 'focal'

    Returns:
        loss_fn(logits: Tensor, targets: Tensor) -> scalar Tensor
    """
    if loss_type == 'ce':
        return lambda logits, targets: F.cross_entropy(logits, targets)

    elif loss_type == 'label_smooth':
        smoothing = float(kwargs.get('smoothing', 0.1))
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        return lambda logits, targets: F.cross_entropy(
            logits, targets, label_smoothing=smoothing
        )

    elif loss_type == 'focal':
        gamma = float(kwargs.get('gamma', 1.0))
        if gamma < 0:
            raise ValueError(f"gamma must be ≥ 0, got {gamma}")
        return FocalLoss(gamma=gamma)

    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            "Choose from: 'ce', 'label_smooth', 'focal'."
        )
