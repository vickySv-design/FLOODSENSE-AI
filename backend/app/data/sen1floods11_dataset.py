"""
FloodSense-AI: Official Sen1Floods11 v1.1 PyTorch Dataset Loader
================================================================
Implements verified multi-modal dataset loading, spatial chip pairing,
sensor radiometric scaling, NoData-safe target label construction,
and reproducible patch generation for Sen1Floods11 v1.1.

Source Reference:
    Bonafilia, D., Tellman, B., Anderson, T., Issifu, A. (2020)
    "Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood
    Algorithms for Sentinel-1." IEEE CVPR Workshops, 2020.
    CC BY 4.0 License.

Semantics & Target Schema:
    Optical: Sentinel-2 Level-1C TOA DNs, 4 bands [B4, B3, B2, B8] via indices [3, 2, 1, 7], scaled DN/10000.
    SAR:     Sentinel-1 C-Band VV & VH in dB, normalized with NaN swath margin masking.
    Target:  3-Class Semantic Target + Ignore Index 255:
             - Class 0: Dry Land (LabelHand=0, including 113,301 disagreement pixels)
             - Class 1: Permanent Water (LabelHand=1 & JRCWaterHand=1)
             - Class 2: Water outside JRC reference (LabelHand=1 & JRCWaterHand=0)
             - Class 255: NoData / Cloud / Invalid SAR margin (dynamically masked)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset


class Sen1Floods11Dataset(Dataset):
    """
    Official PyTorch Dataset for Sen1Floods11 v1.1 hand-labeled ground truth.
    
    Pairs Sentinel-1 SAR (VV, VH) and Sentinel-2 Optical (B4, B3, B2, B8) with
    hydrologically reconciled 3-class segmentation targets.
    """

    OPTICAL_INDICES = [3, 2, 1, 7]  # B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)
    OPTICAL_SCALE = 10000.0         # Level-1C TOA DN scale factor
    IGNORE_INDEX = 255              # Standard PyTorch NoData sentinel

    SPLIT_FILES = {
        "train": "flood_train_data.csv",
        "valid": "flood_valid_data.csv",
        "test": "flood_test_data.csv",
        "bolivia": "flood_bolivia_data.csv"
    }

    def __init__(
        self,
        root_dir: Union[str, Path] = "backend/data/official_sen1floods11",
        split: str = "train",
        patch_size: Optional[int] = 256,
        filter_all_nodata: bool = True,
        nodata_threshold: float = 0.99,
        sar_norm_mode: str = "standard",
        augment: bool = False,
        preload_cache: bool = False
    ):
        """
        Args:
            root_dir: Path to official_sen1floods11 directory.
            split: One of 'train', 'valid', 'test', 'bolivia', or custom path to CSV.
            patch_size: Desired patch size in pixels (e.g. 256). If None or 512, full chip is used.
            filter_all_nodata: Whether to discard patches consisting exclusively of NoData (255).
            nodata_threshold: Maximum allowed fraction of NoData pixels in a patch (default 0.99).
            sar_norm_mode: 'standard' (z-score clip [-3, 3]) or 'linear' ([-28, 0] dB to [0, 1]).
            augment: Whether to apply random spatial flips and 90-degree rotations (train only).
            preload_cache: Whether to preload raster arrays in memory (useful if memory permits).
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.patch_size = patch_size
        self.filter_all_nodata = filter_all_nodata
        self.nodata_threshold = nodata_threshold
        self.sar_norm_mode = sar_norm_mode
        self.augment = augment and (self.split == "train")
        self.preload_cache = preload_cache

        # Subdirectories
        self.raw_dir = self.root_dir / "raw"
        self.s1_dir = self.raw_dir / "S1Hand"
        self.s2_dir = self.raw_dir / "S2Hand"
        self.label_dir = self.raw_dir / "LabelHand"
        self.jrc_dir = self.raw_dir / "JRCWaterHand"
        self.splits_dir = self.root_dir / "splits"

        # Verify directory structure
        for d in [self.s1_dir, self.s2_dir, self.label_dir, self.jrc_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Required Sen1Floods11 directory missing: {d}")

        # Resolve split CSV file
        if self.split in self.SPLIT_FILES:
            self.csv_path = self.splits_dir / self.SPLIT_FILES[self.split]
        else:
            self.csv_path = Path(split)
            if not self.csv_path.exists():
                raise FileNotFoundError(f"Split file not found: {self.csv_path}")

        # Load chip IDs and verify pairing
        self.chip_ids = self._load_split_chip_ids(self.csv_path)
        self.paired_chips = self._verify_and_pair_chips(self.chip_ids)

        # Build patch index
        self.patches = self._build_patch_index()
        self._cache = {}

    def _load_split_chip_ids(self, csv_path: Path) -> List[str]:
        """Parse official CSV split lines and extract unique chip identifiers."""
        chip_ids = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Line format: Ghana_103272_S1Hand.tif,Ghana_103272_LabelHand.tif
                first_entry = line.split(",")[0].strip()
                chip_id = first_entry.replace("_S1Hand.tif", "").replace(".tif", "")
                chip_ids.append(chip_id)
        return chip_ids

    def _verify_and_pair_chips(self, chip_ids: List[str]) -> List[Dict[str, Path]]:
        """Verify that all four required rasters exist for every chip ID."""
        paired = []
        missing = []
        for cid in chip_ids:
            s1_file = self.s1_dir / f"{cid}_S1Hand.tif"
            s2_file = self.s2_dir / f"{cid}_S2Hand.tif"
            label_file = self.label_dir / f"{cid}_LabelHand.tif"
            jrc_file = self.jrc_dir / f"{cid}_JRCWaterHand.tif"

            if not (s1_file.exists() and s2_file.exists() and label_file.exists() and jrc_file.exists()):
                missing.append(cid)
            else:
                paired.append({
                    "chip_id": cid,
                    "s1": s1_file,
                    "s2": s2_file,
                    "label": label_file,
                    "jrc": jrc_file
                })

        if missing:
            raise FileNotFoundError(
                f"Missing raster components for {len(missing)} chips in {self.split} split. "
                f"First missing: {missing[:3]}"
            )
        return paired

    def _build_patch_index(self) -> List[Dict[str, Union[str, int, Tuple[int, int, int, int]]]]:
        """
        Build index of spatial patches.
        If patch_size is None or 512, each chip produces 1 patch.
        If patch_size is 256, each 512x512 chip is partitioned into 4 non-overlapping quadrants.
        """
        patches = []
        for chip_idx, chip_info in enumerate(self.paired_chips):
            cid = chip_info["chip_id"]

            if self.patch_size is None or self.patch_size >= 512:
                # Full chip evaluation
                patches.append({
                    "chip_idx": chip_idx,
                    "chip_id": cid,
                    "patch_idx": 0,
                    "coords": (0, 512, 0, 512)
                })
            elif self.patch_size == 256:
                # 4 Non-overlapping 256x256 quadrants
                quads = [
                    (0, 256, 0, 256),       # Top-Left
                    (0, 256, 256, 512),     # Top-Right
                    (256, 512, 0, 256),     # Bottom-Left
                    (256, 512, 256, 512)    # Bottom-Right
                ]

                # If filtering pure NoData, inspect label raster
                if self.filter_all_nodata and self.split == "train":
                    with rasterio.open(chip_info["label"]) as src:
                        lbl_full = src.read(1)
                    with rasterio.open(chip_info["s1"]) as src:
                        s1_full = src.read(1)

                    for q_idx, (r0, r1, c0, c1) in enumerate(quads):
                        lbl_sub = lbl_full[r0:r1, c0:c1]
                        s1_sub = s1_full[r0:r1, c0:c1]
                        
                        # Discard patch if entirely NoData (-1) or entirely NaN in SAR
                        nodata_ratio = np.mean((lbl_sub == -1) | np.isnan(s1_sub))
                        if nodata_ratio >= self.nodata_threshold:
                            continue  # Skip pure NoData patch

                        patches.append({
                            "chip_idx": chip_idx,
                            "chip_id": cid,
                            "patch_idx": q_idx,
                            "coords": (r0, r1, c0, c1)
                        })
                else:
                    for q_idx, coords in enumerate(quads):
                        patches.append({
                            "chip_idx": chip_idx,
                            "chip_id": cid,
                            "patch_idx": q_idx,
                            "coords": coords
                        })
            else:
                raise ValueError(f"Unsupported patch_size: {self.patch_size}. Must be 256, 512, or None.")

        return patches

    def _read_chip_rasters(self, chip_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read rasters for a single chip, utilizing memory cache if enabled."""
        if self.preload_cache and chip_idx in self._cache:
            return self._cache[chip_idx]

        chip_info = self.paired_chips[chip_idx]
        with rasterio.open(chip_info["s1"]) as src:
            s1 = src.read()  # (2, 512, 512) float32
        with rasterio.open(chip_info["s2"]) as src:
            s2 = src.read()  # (13, 512, 512) int16
        with rasterio.open(chip_info["label"]) as src:
            label = src.read(1)  # (512, 512) int16
        with rasterio.open(chip_info["jrc"]) as src:
            jrc = src.read(1)  # (512, 512) uint8

        data = (s1, s2, label, jrc)
        if self.preload_cache:
            self._cache[chip_idx] = data
        return data

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        patch_info = self.patches[idx]
        chip_idx = patch_info["chip_idx"]
        chip_id = patch_info["chip_id"]
        patch_idx = patch_info["patch_idx"]
        r0, r1, c0, c1 = patch_info["coords"]

        s1_raw, s2_raw, label_raw, jrc_raw = self._read_chip_rasters(chip_idx)

        # Slice spatial window
        s1_patch = s1_raw[:, r0:r1, c0:c1].copy()
        s2_patch = s2_raw[:, r0:r1, c0:c1].copy()
        label_patch = label_raw[r0:r1, c0:c1].copy()
        jrc_patch = jrc_raw[r0:r1, c0:c1].copy()

        # -------------------------------------------------------------
        # 1. OPTICAL PROCESSING: Level-1C TOA DNs [B4, B3, B2, B8]
        # -------------------------------------------------------------
        # Indices: 3 -> B4 (Red), 2 -> B3 (Green), 1 -> B2 (Blue), 7 -> B8 (NIR)
        optical_selected = s2_patch[self.OPTICAL_INDICES, :, :].astype(np.float32)
        optical_scaled = np.clip(optical_selected / self.OPTICAL_SCALE, 0.0, 1.0)
        optical_clean = np.nan_to_num(optical_scaled, nan=0.0, posinf=1.0, neginf=0.0)

        # -------------------------------------------------------------
        # 2. SAR PROCESSING: VV & VH in dB with NaN Swath Margin Masking
        # -------------------------------------------------------------
        vv = s1_patch[0]
        vh = s1_patch[1]
        sar_nan_mask = np.isnan(vv) | np.isnan(vh) | np.isinf(vv) | np.isinf(vh)

        vv_clean = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=-50.0)
        vh_clean = np.nan_to_num(vh, nan=0.0, posinf=0.0, neginf=-50.0)

        if self.sar_norm_mode == "standard":
            # Robust z-score standardization with hard clipping [-3.0, 3.0]
            # S1 empirical means: VV ≈ -12 dB, VH ≈ -19 dB; std: VV ≈ 6 dB, VH ≈ 5 dB
            vv_norm = np.clip((vv_clean - (-12.0)) / 6.0, -3.0, 3.0)
            vh_norm = np.clip((vh_clean - (-19.0)) / 5.0, -3.0, 3.0)
        elif self.sar_norm_mode == "linear":
            # Linear scaling from [-28 dB, 0 dB] into [0, 1]
            vv_norm = np.clip((vv_clean - (-28.0)) / 28.0, 0.0, 1.0)
            vh_norm = np.clip((vh_clean - (-28.0)) / 28.0, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown sar_norm_mode: {self.sar_norm_mode}")

        sar_tensor_np = np.stack([vv_norm, vh_norm], axis=0).astype(np.float32)
        sar_tensor_np[:, sar_nan_mask] = 0.0  # Zero out missing margin pixels

        # -------------------------------------------------------------
        # 3. TARGET RECONCILIATION: 3-Class Mapping + Ignore Index 255
        # -------------------------------------------------------------
        # Class 0: Dry Land (LabelHand=0, including 113,301 disagreement pixels)
        # Class 1: Permanent Water (LabelHand=1 & JRCWaterHand=1)
        # Class 2: Water identified by LabelHand outside JRC reference (LabelHand=1 & JRCWaterHand=0)
        # Class 255: NoData / Cloud (LabelHand=-1) or invalid SAR swath margins
        target = np.full(label_patch.shape, self.IGNORE_INDEX, dtype=np.int64)

        # Apply rule: LabelHand == 0 -> Class 0 (Dry Land)
        target[label_patch == 0] = 0

        # Apply rule: LabelHand == 1 & JRCWaterHand == 1 -> Class 1 (Permanent Water)
        target[(label_patch == 1) & (jrc_patch == 1)] = 1

        # Apply rule: LabelHand == 1 & JRCWaterHand == 0 -> Class 2 (Water outside JRC)
        target[(label_patch == 1) & (jrc_patch == 0)] = 2

        # Enforce NoData: LabelHand == -1 -> 255
        target[label_patch == -1] = self.IGNORE_INDEX

        # Propagate SAR swath margin NaNs to NoData (do not compute loss on missing radar)
        target[sar_nan_mask] = self.IGNORE_INDEX

        # -------------------------------------------------------------
        # 4. DATA AUGMENTATION (Training mode only)
        # -------------------------------------------------------------
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                optical_clean = np.flip(optical_clean, axis=2).copy()
                sar_tensor_np = np.flip(sar_tensor_np, axis=2).copy()
                target = np.flip(target, axis=1).copy()

            # Random vertical flip
            if np.random.rand() > 0.5:
                optical_clean = np.flip(optical_clean, axis=1).copy()
                sar_tensor_np = np.flip(sar_tensor_np, axis=1).copy()
                target = np.flip(target, axis=0).copy()

            # Random 90-degree rotations
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                optical_clean = np.rot90(optical_clean, k=rot_k, axes=(1, 2)).copy()
                sar_tensor_np = np.rot90(sar_tensor_np, k=rot_k, axes=(1, 2)).copy()
                target = np.rot90(target, k=rot_k, axes=(0, 1)).copy()

        return {
            "optical": torch.from_numpy(optical_clean),   # (4, H, W) float32
            "sar": torch.from_numpy(sar_tensor_np),       # (2, H, W) float32
            "target": torch.from_numpy(target),           # (H, W) int64
            "chip_id": chip_id,
            "patch_idx": patch_idx,
            "coords": torch.tensor(r0, dtype=torch.int32)
        }


def get_sen1floods11_dataloader(
    root_dir: Union[str, Path] = "backend/data/official_sen1floods11",
    split: str = "train",
    batch_size: int = 4,
    patch_size: Optional[int] = 256,
    filter_all_nodata: bool = True,
    sar_norm_mode: str = "standard",
    augment: Optional[bool] = None,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Factory helper to construct an official Sen1Floods11 PyTorch DataLoader.
    
    Args:
        root_dir: Dataset root directory.
        split: 'train', 'valid', 'test', or 'bolivia'.
        batch_size: Batch size (default: 4 for 8GB VRAM).
        patch_size: 256 for patched training/evaluation, None for full 512x512 chips.
        filter_all_nodata: Filter pure NoData patches during training.
        sar_norm_mode: 'standard' or 'linear'.
        augment: Data augmentation flag (defaults to True for train, False otherwise).
        shuffle: Whether to shuffle (defaults to True for train, False otherwise).
        num_workers: PyTorch DataLoader worker processes (default 0 for Windows safety).
        pin_memory: Pin memory for high-speed host-to-device CUDA transfer.
    """
    is_train = (split.lower() == "train")
    if augment is None:
        augment = is_train
    if shuffle is None:
        shuffle = is_train

    dataset = Sen1Floods11Dataset(
        root_dir=root_dir,
        split=split,
        patch_size=patch_size,
        filter_all_nodata=filter_all_nodata,
        sar_norm_mode=sar_norm_mode,
        augment=augment
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=is_train
    )

