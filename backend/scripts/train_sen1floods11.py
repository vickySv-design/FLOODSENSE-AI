"""
Academic Training Pipeline for FloodSense-AI on Sen1Floods11 / Multi-Modal SAR-Optical Dataset.
Supports both real Sen1Floods11 directory structures and automatic synthetic multi-modal
dataset generation for offline validation and B.E. Final Year Capstone viva demonstrations.

Usage:
    python scripts/train_sen1floods11.py --epochs 15 --batch_size 4 --lr 0.0005
"""
import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import DEVICE, MODEL_WEIGHTS_PATH, NUM_CLASSES
from app.models.attention_unet import DualStreamAttentionUNet
from app.models.losses import CombinedFloodLoss
from scripts.init_demo_data import generate_synthetic_multimodal_tile


class Sen1Floods11MultiModalDataset(Dataset):
    """
    PyTorch Dataset for paired Sentinel-2 Pre-Flood Optical [4, 256, 256],
    Sentinel-1 Post-Flood SAR [2, 256, 256], and 3-class Ground Truth Mask [256, 256].
    Automatically falls back to procedural multi-modal flood scene synthesis if the
    external 50GB Sen1Floods11 directory is not present on disk.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        synthetic_length: int = 24,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        split_dir = self.data_dir / split
        opt_files = sorted((split_dir / "optical").glob("*.npy")) if (split_dir / "optical").exists() else []

        if len(opt_files) > 0:
            for opt_f in opt_files:
                sar_f = split_dir / "sar" / opt_f.name
                gt_f = split_dir / "labels" / opt_f.name
                if sar_f.exists() and gt_f.exists():
                    self.samples.append(
                        (
                            np.load(opt_f).astype(np.float32),
                            np.load(sar_f).astype(np.float32),
                            np.load(gt_f).astype(np.int64),
                        )
                    )
        else:
            styles = ["chennai", "assam", "wayanad"]
            base_seed = 1000 if split == "train" else 5000
            for i in range(synthetic_length):
                style = styles[i % len(styles)]
                opt_4ch, sar_2ch, gt_mask = generate_synthetic_multimodal_tile(
                    preset_id=style,
                    seed=base_seed + i * 17,
                    cloud_cover_pct=float((i * 7) % 35),
                )
                self.samples.append((opt_4ch, sar_2ch, gt_mask.astype(np.int64)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        opt, sar, gt = self.samples[idx]
        if self.split == "train" and idx % 2 == 1:
            # Random horizontal flip augmentation
            opt = np.flip(opt, axis=2).copy()
            sar = np.flip(sar, axis=2).copy()
            gt = np.flip(gt, axis=1).copy()

        return (
            torch.from_numpy(opt),
            torch.from_numpy(sar),
            torch.from_numpy(gt).long(),
        )


def evaluate_epoch_metrics(
    preds: np.ndarray, targets: np.ndarray, num_classes: int = NUM_CLASSES
) -> Dict[str, float]:
    """Computes macro-averaged mIoU, F1/Dice Score, Precision, and Recall across all classes."""
    ious, f1s, precs, recs = [], [], [], []
    for c in range(num_classes):
        p = preds == c
        t = targets == c
        tp = float(np.logical_and(p, t).sum())
        fp = float(np.logical_and(p, ~t).sum())
        fn = float(np.logical_and(~p, t).sum())

        iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
        prec = (tp + 1e-6) / (tp + fp + 1e-6)
        rec = (tp + 1e-6) / (tp + fn + 1e-6)
        f1 = (2.0 * prec * rec) / (prec + rec + 1e-6)

        ious.append(iou)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    return {
        "miou": float(np.mean(ious)),
        "f1": float(np.mean(f1s)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
    }


def train_pipeline(
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    save_path: Path,
) -> None:
    print("=" * 78)
    print("  FLOODSENSE-AI // SEN1FLOODS11 MULTI-MODAL TRAINING PIPELINE")
    print(f"  Device: {DEVICE} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("  Loss:   CombinedFloodLoss (0.5 * MultiClassFocalLoss + 0.5 * MultiClassDiceLoss)")
    print("=" * 78)

    train_ds = Sen1Floods11MultiModalDataset(Path(data_dir), split="train", synthetic_length=16)
    val_ds = Sen1Floods11MultiModalDataset(Path(data_dir), split="val", synthetic_length=6)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DualStreamAttentionUNet(
        optical_channels=4, sar_channels=2, num_classes=NUM_CLASSES
    ).to(DEVICE)
    criterion = CombinedFloodLoss(num_classes=NUM_CLASSES, focal_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)

    best_miou = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss_accum = 0.0

        for opt_b, sar_b, gt_b in train_loader:
            opt_b = opt_b.to(DEVICE)
            sar_b = sar_b.to(DEVICE)
            gt_b = gt_b.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(opt_b, sar_b)
            loss = criterion(logits, gt_b)
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.item())

        scheduler.step()
        avg_train_loss = train_loss_accum / max(len(train_loader), 1)

        # Validation pass
        model.eval()
        val_loss_accum = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for opt_b, sar_b, gt_b in val_loader:
                opt_b = opt_b.to(DEVICE)
                sar_b = sar_b.to(DEVICE)
                gt_b = gt_b.to(DEVICE)
                logits, _ = model(opt_b, sar_b)
                v_loss = criterion(logits, gt_b)
                val_loss_accum += float(v_loss.item())

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(gt_b.cpu().numpy())

        avg_val_loss = val_loss_accum / max(len(val_loader), 1)
        metrics = evaluate_epoch_metrics(
            np.concatenate(all_preds, axis=0),
            np.concatenate(all_targets, axis=0),
        )
        elapsed = time.perf_counter() - t0
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"TrainLoss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | "
            f"mIoU: {metrics['miou']:.4f} | F1/Dice: {metrics['f1']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
            f"LR: {cur_lr:.6f} ({elapsed:.1f}s)"
        )

        if metrics["miou"] >= best_miou:
            best_miou = metrics["miou"]
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_miou": best_miou,
                    "metrics": metrics,
                },
                save_path,
            )

    print("=" * 78)
    print(f"  [COMPLETE] Best Validation mIoU: {best_miou:.4f} | Saved to: {save_path}")
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Dual-Stream Siamese Attention U-Net on Sen1Floods11"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(BACKEND_ROOT / "data" / "sen1floods11"),
        help="Path to Sen1Floods11 dataset directory (auto-synthesizes if absent)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=8e-4, help="Initial AdamW learning rate")
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(MODEL_WEIGHTS_PATH),
        help="Output checkpoint path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=Path(args.save_path),
    )

