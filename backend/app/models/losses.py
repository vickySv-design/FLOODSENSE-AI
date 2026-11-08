"""NoData-Safe Hybrid Loss Functions for Multi-Modal Flood Segmentation.

Implements:
- MultiClassFocalLoss with ignore_index=255 and class weights
- MultiClassDiceLoss with ignore_index=255 and soft intersection
- CombinedFloodLoss: 0.5 * Focal + 0.5 * Dice
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_target_labels(targets: torch.Tensor, num_classes: int, ignore_index: int) -> None:
    """Validates that all target labels belong to {0..num_classes-1, ignore_index}."""
    invalid_mask = (targets < 0) | ((targets >= num_classes) & (targets != ignore_index))
    if invalid_mask.any():
        invalid_vals = targets[invalid_mask].unique().tolist()
        raise ValueError(
            f"Target contains invalid class labels: {invalid_vals}. "
            f"Allowed classes are 0..{num_classes - 1} and ignore_index={ignore_index}."
        )


class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for addressing extreme foreground-background class imbalance
    between Dry Land, Permanent Water, and Flash Flood Inundation pixels.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Excludes pixels matching ignore_index (default 255) from both loss and gradient.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[torch.Tensor, List[float], Tuple[float, ...]]] = None,
        reduction: str = "mean",
        ignore_index: int = 255,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] unnormalized class scores
            targets: [B, H, W] ground truth class indices in {0..C-1, ignore_index}
        """
        _validate_target_labels(targets, self.num_classes, self.ignore_index)

        valid_mask = (targets != self.ignore_index)
        num_valid = valid_mask.sum()
        if num_valid == 0:
            return (logits * 0.0).sum()

        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        ce_loss = F.cross_entropy(logits, targets_safe, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets_safe.view(-1)).view_as(targets_safe)
            loss = alpha_t * loss

        # Zero out loss at invalid pixels so they contribute zero loss and zero gradient
        loss = loss * valid_mask.float()

        if self.reduction == "mean":
            return loss.sum() / (num_valid.float() + 1e-8)
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiClassDiceLoss(nn.Module):
    """
    Soft Multiclass Dice Loss for crisp flood boundary delineation.
    Excludes pixels matching ignore_index (default 255) from intersection and cardinality.

    Dice = (2 * sum(P * G) + smooth) / (sum(P^2) + sum(G^2) + smooth)
    """

    def __init__(
        self,
        num_classes: int = 3,
        smooth: float = 1e-5,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] unnormalized class scores
            targets: [B, H, W] ground truth class indices in {0..C-1, ignore_index}
        """
        _validate_target_labels(targets, self.num_classes, self.ignore_index)

        valid_mask = (targets != self.ignore_index)
        num_valid = valid_mask.sum()
        if num_valid == 0:
            return (logits * 0.0).sum()

        probs = F.softmax(logits, dim=1)
        valid_mask_exp = valid_mask.unsqueeze(1).float()
        probs_valid = probs * valid_mask_exp

        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        targets_one_hot = (
            F.one_hot(targets_safe.long(), num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        targets_one_hot = targets_one_hot * valid_mask_exp

        dims = (0, 2, 3)
        intersection = torch.sum(probs_valid * targets_one_hot, dims)
        cardinality = torch.sum(probs_valid * probs_valid + targets_one_hot * targets_one_hot, dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CombinedFloodLoss(nn.Module):
    """
    Hybrid Objective Function: L_total = w_focal * L_Focal + w_dice * L_Dice
    Default weights: 0.5 * FocalLoss + 0.5 * DiceLoss
    Default class weights: [0.20, 0.35, 0.45] for Dry Land, Permanent Water, Flood Inundation.
    """

    def __init__(
        self,
        num_classes: int = 3,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        gamma: float = 2.0,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if class_weights is None:
            class_weights = torch.tensor([0.20, 0.35, 0.45], dtype=torch.float32)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.focal = MultiClassFocalLoss(
            gamma=gamma,
            alpha=class_weights,
            ignore_index=ignore_index,
            num_classes=num_classes,
        )
        self.dice = MultiClassDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_f = self.focal(logits, targets)
        loss_d = self.dice(logits, targets)
        return self.focal_weight * loss_f + self.dice_weight * loss_d
