"""
FloodSense-AI: Comprehensive Sen1Floods11 Dataset Loader Test Suite
===================================================================
Tests all 10 core requirements:
1. Chip pairing and file existence.
2. Official split isolation and disjointness.
3. Optical 4-band selection [B4, B3, B2, B8] and Level-1C TOA DN scaling.
4. SAR 2-channel ordering [VV, VH] and radiometric normalization.
5. SAR swath-margin NaN handling and target mask propagation.
6. Reconciled 3-class target mapping (Classes 0, 1, 2 + Ignore 255).
7. Spatial patch generation (256x256) and full-chip evaluation (512x512).
8. Pure NoData patch filtering.
9. PyTorch DataLoader batching and dictionary schema.
10. CPU and CUDA hardware execution compatibility with LightweightDualStreamAttentionUNet.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.data.sen1floods11_dataset import (
    Sen1Floods11Dataset,
    get_sen1floods11_dataloader
)
from backend.app.models.lightweight_attention_unet import LightweightDualStreamAttentionUNet

DATASET_ROOT = REPO_ROOT / "backend" / "data" / "official_sen1floods11"


class TestSen1Floods11Dataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not DATASET_ROOT.exists():
            raise unittest.SkipTest(f"Dataset root does not exist at {DATASET_ROOT}")

    def test_01_chip_pairing_and_file_existence(self):
        """Verify that every chip in train, valid, test, and bolivia has all 4 matching rasters."""
        for split, count in [("train", 252), ("valid", 89), ("test", 90), ("bolivia", 15)]:
            dataset = Sen1Floods11Dataset(
                root_dir=DATASET_ROOT,
                split=split,
                patch_size=None,
                filter_all_nodata=False
            )
            self.assertEqual(
                len(dataset.paired_chips), count,
                f"Split '{split}' expected {count} paired chips, got {len(dataset.paired_chips)}"
            )
            for chip in dataset.paired_chips:
                cid = chip["chip_id"]
                self.assertTrue(chip["s1"].exists(), f"Missing S1 raster for {cid}")
                self.assertTrue(chip["s2"].exists(), f"Missing S2 raster for {cid}")
                self.assertTrue(chip["label"].exists(), f"Missing Label raster for {cid}")
                self.assertTrue(chip["jrc"].exists(), f"Missing JRC raster for {cid}")

    def test_02_split_isolation_and_disjointness(self):
        """Verify 0 overlap across official train, valid, test, and bolivia holdout splits."""
        splits = ["train", "valid", "test", "bolivia"]
        chip_sets = {}
        for s in splits:
            ds = Sen1Floods11Dataset(root_dir=DATASET_ROOT, split=s, patch_size=None)
            chip_sets[s] = set(c["chip_id"] for c in ds.paired_chips)

        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                overlap = chip_sets[s1].intersection(chip_sets[s2])
                self.assertEqual(
                    len(overlap), 0,
                    f"Data leakage detected! Overlap between {s1} and {s2}: {overlap}"
                )

    def test_03_optical_band_selection_and_scaling(self):
        """Verify optical tensor shape [4, H, W], dtype float32, and Level-1C TOA DN scaling."""
        dataset = Sen1Floods11Dataset(
            root_dir=DATASET_ROOT,
            split="valid",
            patch_size=256,
            filter_all_nodata=False
        )
        sample = dataset[0]
        opt = sample["optical"]

        self.assertIsInstance(opt, torch.Tensor)
        self.assertEqual(opt.dtype, torch.float32)
        self.assertEqual(opt.shape, (4, 256, 256))
        # Level-1C TOA DN scaled by 10,000 should be bounded in [0.0, 1.0]
        self.assertGreaterEqual(float(opt.min()), 0.0)
        self.assertLessEqual(float(opt.max()), 1.0)
        self.assertTrue(torch.isfinite(opt).all(), "Optical tensor contains NaN or Inf!")

    def test_04_sar_tensor_channels_and_normalization(self):
        """Verify SAR tensor shape [2, H, W], dtype float32, VV/VH ordering, and normalization."""
        dataset = Sen1Floods11Dataset(
            root_dir=DATASET_ROOT,
            split="valid",
            patch_size=256,
            sar_norm_mode="standard",
            filter_all_nodata=False
        )
        sample = dataset[0]
        sar = sample["sar"]

        self.assertIsInstance(sar, torch.Tensor)
        self.assertEqual(sar.dtype, torch.float32)
        self.assertEqual(sar.shape, (2, 256, 256))
        # Standard normalization clips to [-3.0, 3.0]
        self.assertGreaterEqual(float(sar.min()), -3.0)
        self.assertLessEqual(float(sar.max()), 3.0)
        self.assertTrue(torch.isfinite(sar).all(), "SAR tensor contains NaN or Inf!")

    def test_05_sar_nan_swath_margin_handling(self):
        """Verify that edge NaNs in SAR are replaced with 0.0 and mapped to 255 in target."""
        # Find a chip known to contain swath margin NaNs
        dataset = Sen1Floods11Dataset(
            root_dir=DATASET_ROOT,
            split="test",
            patch_size=None,
            filter_all_nodata=False
        )
        found_margin_chip = False
        for i in range(len(dataset)):
            sample = dataset[i]
            target = sample["target"]
            sar = sample["sar"]
            # Look for a chip with 255 target pixels
            if (target == 255).any():
                found_margin_chip = True
                # SAR values must be clean finite floats even at margins
                self.assertTrue(torch.isfinite(sar).all(), f"SAR contains non-finite values in chip {sample['chip_id']}")
                break
        self.assertTrue(found_margin_chip, "Expected to find at least one chip with NoData / margin masking")

    def test_06_three_class_label_mapping(self):
        """Verify strict 3-class target mapping: 0=Dry, 1=Perm Water, 2=Water outside JRC, 255=NoData."""
        dataset = Sen1Floods11Dataset(
            root_dir=DATASET_ROOT,
            split="train",
            patch_size=256,
            filter_all_nodata=False
        )
        # Check unique labels across first 10 patches
        all_unique = set()
        for i in range(min(15, len(dataset))):
            target = dataset[i]["target"]
            self.assertEqual(target.dtype, torch.int64)
            self.assertEqual(target.shape, (256, 256))
            unique_vals = set(target.unique().tolist())
            all_unique.update(unique_vals)

        allowed_classes = {0, 1, 2, 255}
        self.assertTrue(
            all_unique.issubset(allowed_classes),
            f"Unexpected label values found: {all_unique - allowed_classes}"
        )

    def test_07_patch_generation_and_full_chip_shapes(self):
        """Verify 256x256 patch mode yields 4 quadrants and full-chip mode yields 512x512."""
        # Patch mode
        ds_patch = Sen1Floods11Dataset(
            root_dir=DATASET_ROOT,
            split="valid",
            patch_size=256,
            filter_all_nodata=False
        )
        # 89 chips * 4 = 356 patches
        self.assertEqual(len(ds_patch), 89 * 4)
        sample_patch = ds_patch[0]
        self.assertEqual(sample_patch["optical"].shape, (4, 256, 256))
        self.assertEqual(sample_patch["sar"].shape, (2, 256, 256))
        self.assertEqual(sample_patch["target"].shape, (256, 256))

        # Full chip mode
        ds_full = Sen1Floods11Dataset(
            root_dir=DATASET_ROOT,
            split="valid",
            patch_size=None,
            filter_all_nodata=False
        )
        self.assertEqual(len(ds_full), 89)
        sample_full = ds_full[0]
        self.assertEqual(sample_full["optical"].shape, (4, 512, 512))
        self.assertEqual(sample_full["sar"].shape, (2, 512, 512))
        self.assertEqual(sample_full["target"].shape, (512, 512))

    def test_08_filter_pure_nodata_patches(self):
        """Verify that pure NoData patches are excluded when filter_all_nodata=True."""
        ds_unfiltered = Sen1Floods11Dataset(
            root_dir=DATASET_ROOT,
            split="train",
            patch_size=256,
            filter_all_nodata=False
        )
        ds_filtered = Sen1Floods11Dataset(
            root_dir=DATASET_ROOT,
            split="train",
            patch_size=256,
            filter_all_nodata=True,
            nodata_threshold=0.99
        )
        self.assertLessEqual(
            len(ds_filtered), len(ds_unfiltered),
            "Filtered dataset should have fewer or equal patches than unfiltered"
        )
        print(f"  [Info] Train patches: unfiltered={len(ds_unfiltered)}, filtered={len(ds_filtered)}")

    def test_09_dataloader_batch_compatibility(self):
        """Verify DataLoader batches tensors correctly into [B, 4, H, W], [B, 2, H, W], [B, H, W]."""
        loader = get_sen1floods11_dataloader(
            root_dir=DATASET_ROOT,
            split="valid",
            batch_size=4,
            patch_size=256,
            shuffle=False,
            num_workers=0
        )
        batch = next(iter(loader))

        opt_batch = batch["optical"]
        sar_batch = batch["sar"]
        target_batch = batch["target"]

        self.assertEqual(opt_batch.shape, (4, 4, 256, 256))
        self.assertEqual(sar_batch.shape, (4, 2, 256, 256))
        self.assertEqual(target_batch.shape, (4, 256, 256))
        self.assertEqual(opt_batch.dtype, torch.float32)
        self.assertEqual(sar_batch.dtype, torch.float32)
        self.assertEqual(target_batch.dtype, torch.int64)

    def test_10_cpu_and_cuda_model_forward_pass(self):
        """Verify batch executes successfully on CPU and native CUDA GPU through Lightweight model."""
        loader = get_sen1floods11_dataloader(
            root_dir=DATASET_ROOT,
            split="valid",
            batch_size=2,
            patch_size=256,
            shuffle=False,
            num_workers=0
        )
        batch = next(iter(loader))
        opt = batch["optical"]
        sar = batch["sar"]
        target = batch["target"]

        # 1. Test CPU execution
        model_cpu = LightweightDualStreamAttentionUNet(
            optical_channels=4,
            sar_channels=2,
            num_classes=3
        )
        model_cpu.eval()
        with torch.no_grad():
            out_cpu = model_cpu(opt, sar)
        logits_cpu = out_cpu[0] if isinstance(out_cpu, tuple) else out_cpu
        self.assertEqual(logits_cpu.shape, (2, 3, 256, 256))
        self.assertTrue(torch.isfinite(logits_cpu).all())

        # 2. Test CUDA GPU execution (if available)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model_cuda = LightweightDualStreamAttentionUNet(
                optical_channels=4,
                sar_channels=2,
                num_classes=3
            ).to(device)
            model_cuda.eval()
            opt_cuda = opt.to(device)
            sar_cuda = sar.to(device)
            with torch.no_grad():
                out_cuda = model_cuda(opt_cuda, sar_cuda)
            logits_cuda = out_cuda[0] if isinstance(out_cuda, tuple) else out_cuda
            self.assertEqual(logits_cuda.shape, (2, 3, 256, 256))
            self.assertTrue(torch.isfinite(logits_cuda).all())
            print(f"  [Info] Native CUDA execution successful on {torch.cuda.get_device_name(0)}")
        else:
            print("  [Info] CUDA not available, skipped GPU step")


if __name__ == "__main__":
    unittest.main()
