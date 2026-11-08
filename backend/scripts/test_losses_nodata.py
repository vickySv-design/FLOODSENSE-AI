"""Unit Test Suite for NoData-Safe Hybrid Flood Loss Functions.

Tests:
1. Normal targets containing only classes 0, 1, and 2.
2. Targets containing a mixture of 0, 1, 2, and 255.
3. An image containing only NoData (255) pixels.
4. A batch containing one valid image and one all-NoData image.
5. Invalid class values (such as 3, 4, 254, -1) raising ValueError.
6. Correct loss output shape (0-dim scalar) and positive loss values.
7. Finite gradients across all parameters and input tensors.
8. NoData pixels having zero contribution to loss and gradients (compared to masked reference).
9. Correct CPU execution behavior.
10. Correct CUDA execution behavior when CUDA is available (or marked appropriately if CPU-only).
"""

import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from app.models.losses import CombinedFloodLoss, MultiClassDiceLoss, MultiClassFocalLoss


class TestNoDataLossFunctions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_classes = 3
        self.ignore_index = 255
        self.criterion = CombinedFloodLoss(
            num_classes=self.num_classes,
            focal_weight=0.5,
            dice_weight=0.5,
            ignore_index=self.ignore_index,
        )

    def test_01_normal_targets_classes_0_1_2(self):
        """Test 1: Normal targets containing only classes 0, 1, and 2."""
        logits = torch.randn(2, 3, 32, 32, requires_grad=True)
        targets = torch.randint(0, 3, (2, 32, 32), dtype=torch.long)

        loss = self.criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(loss.dim(), 0)

        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        print("  [PASS] Test 1: Normal targets (0, 1, 2) produce positive scalar loss with finite gradients.")

    def test_02_mixture_targets_with_nodata_255(self):
        """Test 2: Targets containing a mixture of 0, 1, 2, and 255."""
        logits = torch.randn(2, 3, 32, 32, requires_grad=True)
        targets = torch.randint(0, 3, (2, 32, 32), dtype=torch.long)
        # Inject NoData 255 into 30% of pixels
        nodata_mask = torch.rand(2, 32, 32) < 0.30
        targets[nodata_mask] = 255

        loss = self.criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        # Gradients at NoData pixels must be exactly zero
        nodata_grads = logits.grad[nodata_mask.unsqueeze(1).expand_as(logits)]
        self.assertEqual(nodata_grads.abs().sum().item(), 0.0)
        print("  [PASS] Test 2: Mixture targets with 255 execute without crashing; 255 pixels have 0.0 gradient.")

    def test_03_all_nodata_image(self):
        """Test 3: An image containing 100% NoData (255) pixels."""
        logits = torch.randn(1, 3, 32, 32, requires_grad=True)
        targets = torch.full((1, 32, 32), 255, dtype=torch.long)

        loss = self.criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)

        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad.abs().sum().item(), 0.0)
        print("  [PASS] Test 3: 100% NoData image produces exactly 0.0 loss and 0.0 gradients without NaN/Inf.")

    def test_04_batch_with_valid_and_all_nodata_images(self):
        """Test 4: A batch containing one valid image and one all-NoData image."""
        logits = torch.randn(2, 3, 32, 32, requires_grad=True)
        targets = torch.randint(0, 3, (2, 32, 32), dtype=torch.long)
        # Make the second sample 100% NoData
        targets[1, :, :] = 255

        loss = self.criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

        # Sample 0 (valid) must have non-zero gradients
        self.assertGreater(logits.grad[0].abs().sum().item(), 0.0)
        # Sample 1 (all NoData) must have strictly zero gradients
        self.assertEqual(logits.grad[1].abs().sum().item(), 0.0)
        print("  [PASS] Test 4: Batch of [valid, all-NoData] computes loss only from valid image.")

    def test_05_invalid_class_labels_rejected(self):
        """Test 5: Invalid class values such as 3, 4, 254, -1 raise ValueError."""
        logits = torch.randn(1, 3, 16, 16)

        for invalid_val in [3, 4, 5, 254, -1]:
            targets = torch.zeros(1, 16, 16, dtype=torch.long)
            targets[0, 5, 5] = invalid_val

            with self.assertRaises(ValueError, msg=f"Should raise ValueError for label {invalid_val}"):
                self.criterion(logits, targets)

        print("  [PASS] Test 5: Invalid class labels (3, 4, 254, -1) correctly rejected with ValueError.")

    def test_06_correct_loss_output_shape_and_scalar(self):
        """Test 6: Correct loss output shape (0-dim scalar) and positive loss values."""
        logits = torch.randn(4, 3, 64, 64)
        targets = torch.randint(0, 3, (4, 64, 64), dtype=torch.long)

        loss = self.criterion(logits, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0.0)
        print("  [PASS] Test 6: Loss output is a standard 0-dimensional scalar float.")

    def test_07_finite_gradients_through_model(self):
        """Test 7: Finite gradients through full backward graph with optimizer step."""
        from app.models.lightweight_attention_unet import LightweightDualStreamAttentionUNet

        model = LightweightDualStreamAttentionUNet()
        model.train()
        opt = torch.randn(2, 4, 64, 64)
        sar = torch.randn(2, 2, 64, 64)
        targets = torch.randint(0, 3, (2, 64, 64), dtype=torch.long)
        targets[0, :10, :10] = 255  # Partial NoData

        logits, _ = model(opt, sar)
        loss = self.criterion(logits, targets)
        loss.backward()

        all_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        self.assertTrue(all_finite)
        print("  [PASS] Test 7: Full backward pass through LightweightDualStreamAttentionUNet produces 100% finite gradients.")

    def test_08_nodata_zero_contribution_comparison(self):
        """Test 8: NoData pixels have zero contribution compared to an identically masked calculation."""
        torch.manual_seed(101)
        # Create a 1x3x4x4 logit tensor
        logits_a = torch.randn(1, 3, 4, 4, requires_grad=True)
        logits_b = logits_a.clone().detach().requires_grad_(True)

        # Target has valid pixels in top half, NoData in bottom half
        targets_a = torch.tensor([
            [[0, 1, 2, 1],
             [1, 2, 0, 2],
             [255, 255, 255, 255],
             [255, 255, 255, 255]]
        ], dtype=torch.long)

        # In targets_b, change the NoData region values to other NoData patterns or alter logits in NoData region
        # altering logits_b in the NoData region should NOT change the loss or the gradients in the valid region!
        logits_b_mod = logits_b.clone()
        with torch.no_grad():
            logits_b_mod[:, :, 2:, :] += torch.randn(1, 3, 2, 4) * 10.0  # Massive perturbation in NoData region

        loss_a = self.criterion(logits_a, targets_a)
        loss_b = self.criterion(logits_b_mod, targets_a)

        loss_a.backward()
        loss_b.backward()

        # Losses must be identical despite massive logit perturbations in NoData region
        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=5)

        # Gradients in valid region (rows 0 and 1) must be identical
        valid_grad_a = logits_a.grad[:, :, :2, :]
        valid_grad_b = logits_b.grad[:, :, :2, :]
        self.assertTrue(torch.allclose(valid_grad_a, valid_grad_b, atol=1e-5))

        # Gradients in NoData region (rows 2 and 3) must be exactly zero
        self.assertEqual(logits_a.grad[:, :, 2:, :].abs().sum().item(), 0.0)
        self.assertEqual(logits_b.grad[:, :, 2:, :].abs().sum().item(), 0.0)
        print("  [PASS] Test 8: Perturbations in NoData pixels cause zero delta in loss and zero gradient.")

    def test_09_correct_cpu_behavior(self):
        """Test 9: Correct execution on CPU device."""
        logits = torch.randn(1, 3, 16, 16, device="cpu", requires_grad=True)
        targets = torch.randint(0, 3, (1, 16, 16), device="cpu", dtype=torch.long)
        targets[0, :4, :4] = 255

        loss = self.criterion(logits, targets)
        self.assertEqual(loss.device.type, "cpu")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(logits.grad.device.type, "cpu")
        print("  [PASS] Test 9: Native CPU execution verified.")

    def test_10_correct_cuda_behavior(self):
        """Test 10: Correct execution on CUDA device (or skipped if CUDA not active in this Python)."""
        if not torch.cuda.is_available():
            print("  [SKIPPED] Test 10: CUDA is not available in the current global environment (tested separately in .venv_cuda).")
            self.skipTest("CUDA not available in active environment.")

        device = torch.device("cuda")
        criterion_cuda = CombinedFloodLoss(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        ).to(device)

        logits = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
        targets = torch.randint(0, 3, (2, 32, 32), device=device, dtype=torch.long)
        targets[:, :8, :8] = 255

        loss = criterion_cuda(logits, targets)
        self.assertEqual(loss.device.type, "cuda")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(logits.grad.device.type, "cuda")
        print("  [PASS] Test 10: Native CUDA execution verified on GPU.")


if __name__ == "__main__":
    print("=" * 75)
    print("FLOODSENSE-AI: NODATA-SAFE HYBRID LOSS UNIT TEST SUITE")
    print("=" * 75)
    unittest.main(verbosity=1)
