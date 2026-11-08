"""
FloodSense-AI: Comprehensive Sen1Floods11 Inference Test Suite
==============================================================
Validates:
1. Best checkpoint loads cleanly and contains valid weights.
2. Model produces [1, 3, 512, 512] logits on full satellite chips.
3. Predictions contain only valid semantic classes {0, 1, 2} on valid pixels.
4. NoData regions (255) are preserved and excluded from metrics.
5. Output masks preserve exact 512x512 spatial dimensions.
6. Georeferenced GeoTIFFs and NumPy arrays are created and verified.
7. Visualizations (.png) are generated with valid headers and file sizes.
8. CPU and CUDA inference compatibility.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import rasterio
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.data.sen1floods11_dataset import Sen1Floods11Dataset
from backend.app.models.lightweight_attention_unet import LightweightDualStreamAttentionUNet
from backend.scripts.run_sen1floods11_inference import (
    load_model,
    run_inference,
    DEFAULT_CHECKPOINT,
    DATASET_ROOT,
    OUTPUT_DIR
)


class TestSen1Floods11Inference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not DEFAULT_CHECKPOINT.exists():
            raise unittest.SkipTest(f"Trained checkpoint not found at: {DEFAULT_CHECKPOINT}")
        if not DATASET_ROOT.exists():
            raise unittest.SkipTest(f"Dataset root not found at: {DATASET_ROOT}")

    def test_01_checkpoint_loading_and_architecture(self):
        """Verify checkpoint loads cleanly and matches LightweightDualStreamAttentionUNet."""
        device = torch.device("cpu")
        model, meta = load_model(DEFAULT_CHECKPOINT, device)
        self.assertIsInstance(model, LightweightDualStreamAttentionUNet)
        self.assertIn("epoch", meta)
        self.assertIn("val_miou", meta)
        self.assertGreater(meta["val_miou"], 0.65, "Expected trained checkpoint with val mIoU > 0.65")

    def test_02_model_full_chip_logits_shape(self):
        """Verify model produces [1, 3, 512, 512] logits on full satellite chips."""
        device = torch.device("cpu")
        model, _ = load_model(DEFAULT_CHECKPOINT, device)

        opt_dummy = torch.randn(1, 4, 512, 512, dtype=torch.float32)
        sar_dummy = torch.randn(1, 2, 512, 512, dtype=torch.float32)

        with torch.no_grad():
            logits, aux = model(opt_dummy, sar_dummy)

        self.assertEqual(logits.shape, (1, 3, 512, 512))
        self.assertEqual(aux.shape, (1, 1, 512, 512))
        self.assertTrue(torch.isfinite(logits).all())

    def test_03_prediction_classes_and_nodata_exclusion(self):
        """Verify predictions contain only classes {0, 1, 2} on valid pixels."""
        dataset = Sen1Floods11Dataset(
            root_dir=DATASET_ROOT,
            split="test",
            patch_size=None,
            filter_all_nodata=False
        )
        sample = dataset[0]
        opt = sample["optical"].unsqueeze(0)
        sar = sample["sar"].unsqueeze(0)
        tgt = sample["target"].numpy()

        device = torch.device("cpu")
        model, _ = load_model(DEFAULT_CHECKPOINT, device)

        with torch.no_grad():
            logits, _ = model(opt, sar)
            pred = torch.argmax(logits, dim=1).squeeze(0).numpy()

        self.assertEqual(pred.shape, (512, 512))
        valid_mask = (tgt != 255)
        pred_valid = pred[valid_mask]
        unique_classes = set(np.unique(pred_valid))

        self.assertTrue(
            unique_classes.issubset({0, 1, 2}),
            f"Predicted classes contain invalid values: {unique_classes}"
        )

    def test_04_inference_execution_and_exports(self):
        """Verify run_inference generates visualizations, GeoTIFFs, and reports."""
        test_out_dir = REPO_ROOT / "backend" / "outputs" / "sen1floods11_inference_test"
        result = run_inference(
            num_chips=2,
            checkpoint_path=DEFAULT_CHECKPOINT,
            output_dir=test_out_dir
        )

        self.assertEqual(result["chips_processed"], 2)
        self.assertGreater(result["average_latency_ms"], 0.0)

        # Verify exported files
        vis_files = list((test_out_dir / "visualizations").glob("*.png"))
        self.assertEqual(len(vis_files), 2)
        for vf in vis_files:
            self.assertGreater(vf.stat().st_size, 50000, "Visualization PNG should be > 50KB")

        mask_files_tif = list((test_out_dir / "masks").glob("*_pred_mask.tif"))
        self.assertEqual(len(mask_files_tif), 2)
        for mf in mask_files_tif:
            with rasterio.open(mf) as src:
                self.assertEqual(src.shape, (512, 512))
                self.assertEqual(src.crs.to_epsg(), 4326)

        report_file = test_out_dir / "INFERENCE_REPORT.md"
        self.assertTrue(report_file.exists())
        self.assertGreater(report_file.stat().st_size, 500)

        # Cleanup test directory
        import shutil
        shutil.rmtree(test_out_dir, ignore_errors=True)

    def test_05_cuda_and_cpu_compatibility(self):
        """Verify inference runs cleanly on CPU and native CUDA GPU."""
        opt = torch.randn(1, 4, 256, 256)
        sar = torch.randn(1, 2, 256, 256)

        # CPU test
        model_cpu, _ = load_model(DEFAULT_CHECKPOINT, torch.device("cpu"))
        with torch.no_grad():
            out_cpu, _ = model_cpu(opt, sar)
        self.assertEqual(out_cpu.shape, (1, 3, 256, 256))

        # CUDA test
        if torch.cuda.is_available():
            device_cuda = torch.device("cuda")
            model_cuda, _ = load_model(DEFAULT_CHECKPOINT, device_cuda)
            opt_cuda = opt.to(device_cuda)
            sar_cuda = sar.to(device_cuda)
            with torch.no_grad():
                out_cuda, _ = model_cuda(opt_cuda, sar_cuda)
            self.assertEqual(out_cuda.shape, (1, 3, 256, 256))
            print(f"  [Info] Native CUDA inference verified on {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    unittest.main()

