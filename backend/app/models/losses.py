from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for addressing extreme foreground-background class imbalance
    between Dry Land, Permanent Water, and Flash Flood Inundation pixels.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] unnormalized class scores
            targets: [B, H, W] ground truth class indices in {0..C-1}
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets.view(-1)).view_as(targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiClassDiceLoss(nn.Module):
    """
    Soft Multiclass Dice Loss for crisp flood boundary delineation.

    Dice = (2 * sum(P * G) + smooth) / (sum(P^2) + sum(G^2) + smooth)
    """

    def __init__(self, num_classes: int = 3, smooth: float = 1e-5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = (
            F.one_hot(targets.long(), num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs * probs + targets_one_hot * targets_one_hot, dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CombinedFloodLoss(nn.Module):
    """
    Hybrid Objective Function: L_total = w_focal * L_Focal + w_dice * L_Dice
    Default weights: 0.5 * FocalLoss + 0.5 * DiceLoss
    """

    def __init__(
        self,
        num_classes: int = 3,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        class_weights = torch.tensor([0.20, 0.35, 0.45], dtype=torch.float32)
        self.focal = MultiClassFocalLoss(gamma=gamma, alpha=class_weights)
        self.dice = MultiClassDiceLoss(num_classes=num_classes)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_f = self.focal(logits, targets)
        loss_d = self.dice(logits, targets)
        return self.focal_weight * loss_f + self.dice_weight * loss_d

