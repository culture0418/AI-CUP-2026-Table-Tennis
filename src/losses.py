"""Loss functions.

Two losses:
  - FocalLoss: standard focal loss with uniform label smoothing.
              Used by V25-A on both action and point heads.
  - AsymSpatialFocalLoss: V27 innovation — focal loss with non-uniform label
              smoothing that redistributes mass for `pointId=3` (反手短球, rarest
              0.9% in train) to its 9-zone-grid spatial neighbors class 2 (中間短,
              same row) and class 6 (反手半長, same column). Encodes table tennis
              tactical knowledge "adjacent landing positions are substitutable"
              directly into the loss function.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    LABEL_SMOOTH, N_POINT,
    FOCUS_CLASS, FOCUS_NEIGHBORS, FOCUS_SPATIAL_EPS, FOCUS_UNIFORM_EPS,
)


class FocalLoss(nn.Module):
    """Focal loss with uniform label smoothing. Used by V25-A."""

    def __init__(self, weight=None, gamma=2.0, label_smoothing=LABEL_SMOOTH):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        K = logits.size(-1)
        target_oh = torch.full_like(logits, self.ls / max(K - 1, 1))
        target_oh.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=-1).clamp(min=-30.0)
        focal = (1 - log_p.exp()).pow(self.gamma)
        loss = -focal * log_p
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        return (target_oh * loss).sum(dim=-1).mean()


def build_asym_label_distribution():
    """Build the asymmetric label-smoothing distribution for V27 point head.

    Class 3 (反手短) gets 80% mass + 7.5% to each of {2, 6} spatial neighbors
    + 5%/6 = 0.83% to each of the other 6 classes.
    All non-focus classes use standard uniform label-smoothing (0.10 spread).
    """
    smoothed = np.zeros((N_POINT, N_POINT), dtype=np.float32)
    for c in range(N_POINT):
        if c == FOCUS_CLASS:
            n_nb = len(FOCUS_NEIGHBORS)
            n_other = N_POINT - 1 - n_nb
            smoothed[c, c] = 1.0 - FOCUS_SPATIAL_EPS - FOCUS_UNIFORM_EPS
            for j in range(N_POINT):
                if j == c:
                    continue
                smoothed[c, j] = (FOCUS_SPATIAL_EPS / n_nb) if j in FOCUS_NEIGHBORS \
                                 else (FOCUS_UNIFORM_EPS / n_other)
        else:
            smoothed[c, c] = 1.0 - LABEL_SMOOTH
            for j in range(N_POINT):
                if j != c:
                    smoothed[c, j] = LABEL_SMOOTH / (N_POINT - 1)
    return smoothed


class AsymSpatialFocalLoss(nn.Module):
    """V27 point-head loss — focal loss with TT-domain asymmetric spatial label smoothing.

    For class-3 (反手短球, the rarest 0.9% landing zone) samples, retains 80% mass
    at the true class but redistributes 15% to {class 2, class 6} spatial neighbors
    and the remaining 5% uniformly. For all other classes, falls back to standard
    uniform label smoothing (LABEL_SMOOTH = 0.10).
    """

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer("smoothed", torch.from_numpy(build_asym_label_distribution()))
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        td = self.smoothed[targets]  # (B, N_POINT)
        log_p = F.log_softmax(logits, dim=-1).clamp(min=-30.0)
        focal = (1 - log_p.exp()).pow(self.gamma)
        loss = -focal * log_p
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        return (td * loss).sum(dim=-1).mean()
