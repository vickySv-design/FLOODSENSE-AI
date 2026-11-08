"""
FloodSense-AI: Automated Scientific Integrity & Domain Reconciliation Test Suite
=================================================================================
Phase 4D Test Suite: Verifies mathematical consistency, pixel-level conservation,
geodesic area reconciliation, and cross-artifact synchronization.

Artifacts Audited:
- backend/data/flood_event/before/Bolivia_103757_S1_PreEvent_20180203.tif
- backend/data/official_sen1floods11/raw/S1Hand/Bolivia_103757_S1Hand.tif
- backend/data/official_sen1floods11/raw/LabelHand/Bolivia_103757_LabelHand.tif
- backend/data/official_sen1floods11/raw/JRCWaterHand/Bolivia_103757_JRCWaterHand.tif
- backend/outputs/flood_event_analysis/potential_inundation_mask.tif
- backend/outputs/flood_event_analysis/delta_VV.tif
- backend/outputs/flood_event_analysis/delta_VH.tif
- backend/outputs/flood_event_analysis/change_detection_metrics.json
- backend/outputs/flood_event_analysis/threshold_sensitivity.csv
- backend/outputs/flood_event_analysis/FLOOD_CHANGE_DETECTION_REPORT.md
"""

import os
import sys
import json
import csv
import unittest
import numpy as np
import rasterio

class TestFloodChangeDetectionIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pre_path = "backend/data/flood_event/before/Bolivia_103757_S1_PreEvent_20180203.tif"
        cls.ev_path = "backend/data/official_sen1floods11/raw/S1Hand/Bolivia_103757_S1Hand.tif"
        cls.lbl_path = "backend/data/official_sen1floods11/raw/LabelHand/Bolivia_103757_LabelHand.tif"
        cls.jrc_path = "backend/data/official_sen1floods11/raw/JRCWaterHand/Bolivia_103757_JRCWaterHand.tif"
        cls.mask_path = "backend/outputs/flood_event_analysis/potential_inundation_mask.tif"
        cls.delta_vv_path = "backend/outputs/flood_event_analysis/delta_VV.tif"
        cls.delta_vh_path = "backend/outputs/flood_event_analysis/delta_VH.tif"
        cls.json_path = "backend/outputs/flood_event_analysis/change_detection_metrics.json"
        cls.csv_path = "backend/outputs/flood_event_analysis/threshold_sensitivity.csv"
        cls.report_path = "backend/outputs/flood_event_analysis/FLOOD_CHANGE_DETECTION_REPORT.md"

        # Load rasters
        with rasterio.open(cls.pre_path) as s:
            cls.pre_vv, cls.pre_vh = s.read()
        with rasterio.open(cls.ev_path) as s:
            cls.ev_vv, cls.ev_vh = s.read()
        with rasterio.open(cls.lbl_path) as s:
            cls.lbl = s.read(1)
        with rasterio.open(cls.jrc_path) as s:
            cls.jrc = s.read(1)
        with rasterio.open(cls.mask_path) as s:
            cls.mask = s.read(1)
        with rasterio.open(cls.delta_vv_path) as s:
            cls.delta_vv = s.read(1)

        # Load JSON metrics
        with open(cls.json_path, "r", encoding="utf-8") as f:
            cls.metrics_data = json.load(f)

        # Load CSV sensitivity rows
        cls.csv_rows = []
        with open(cls.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls.csv_rows.append(row)

        # Load Markdown report
        with open(cls.report_path, "r", encoding="utf-8") as f:
            cls.report_text = f.read()

        # Compute ground truth evaluation domains
        cls.valid_sar = (~np.isnan(cls.pre_vv)) & (~np.isnan(cls.ev_vv))
        cls.valid_lbl = (cls.lbl == 0) | (cls.lbl == 1)
        cls.eval_mask = cls.valid_sar & cls.valid_lbl
        cls.annotator_excluded_mask = cls.valid_sar & (~cls.valid_lbl)

    def test_01_domain_pixel_counts_and_decomposition(self):
        """Test that raster domains decompose and sum exactly without loss."""
        total_pixels = 512 * 512  # 262,144
        self.assertEqual(self.pre_vv.size, total_pixels)
        
        # Domain A (Full SAR swath)
        count_sar = int(np.sum(self.valid_sar))
        self.assertEqual(count_sar, 101623, "Domain A must contain exactly 101,623 mutual valid SAR pixels")

        # Domain C (Evaluation domain)
        count_eval = int(np.sum(self.eval_mask))
        self.assertEqual(count_eval, 88077, "Domain C must contain exactly 88,077 evaluation pixels")

        # Region D (Annotator excluded region)
        count_excluded = int(np.sum(self.annotator_excluded_mask))
        self.assertEqual(count_excluded, 13546, "Region D must contain exactly 13,546 excluded pixels")

        # Exact pixel conservation check
        self.assertEqual(count_eval + count_excluded, count_sar, "Domain C + Region D must equal Domain A exactly")

    def test_02_nodata_representation_and_mask_integrity(self):
        """Test that potential_inundation_mask.tif correctly maps 0, 1, and 255."""
        vals, counts = np.unique(self.mask, return_counts=True)
        count_dict = dict(zip(vals.tolist(), counts.tolist()))
        
        self.assertIn(0, count_dict, "Mask must contain dry land pixels (value 0)")
        self.assertIn(1, count_dict, "Mask must contain potential water pixels (value 1)")
        self.assertIn(255, count_dict, "Mask must contain NoData pixels (value 255)")

        # In Domain A, pixels are either 0 or 1
        self.assertEqual(count_dict[0] + count_dict[1], 101623, "Swath pixels (0 and 1) must equal exactly 101,623")
        # Out-of-swath pixels must be 255
        self.assertEqual(count_dict[255], 160521, "Out-of-swath NoData must equal exactly 160,521 pixels")
        self.assertEqual(count_dict[0] + count_dict[1] + count_dict[255], 262144, "Total tile pixels must equal 262,144")

        # Value 1 corresponds to Delta VV < -5.0 dB in SAR swath
        expected_changed_swath = int(np.sum((self.delta_vv < -5.0) & self.valid_sar))
        self.assertEqual(count_dict[1], expected_changed_swath, "Mask value 1 must equal Delta VV < -5.0 in swath (45,788)")
        self.assertEqual(count_dict[1], 45788)

    def test_03_confusion_matrix_reconciliation(self):
        """Test that confusion matrix totals sum to exactly the evaluation domain."""
        target_water = self.eval_mask & (self.lbl == 1) & (self.jrc == 0)
        dry_ref = self.eval_mask & (self.lbl == 0)
        
        pred_eval = (self.delta_vv < -5.0) & self.eval_mask

        tp = int(np.sum(pred_eval & target_water))
        fp = int(np.sum(pred_eval & dry_ref))
        fn = int(np.sum((~pred_eval) & target_water))
        tn = int(np.sum((~pred_eval) & dry_ref))

        self.assertEqual(tp, 29249, "True positives must equal 29,249")
        self.assertEqual(fp, 8276, "False positives must equal 8,276")
        self.assertEqual(fn, 6113, "False negatives must equal 6,113")
        self.assertEqual(tn, 44439, "True negatives must equal 44,439")

        # Sum checks
        self.assertEqual(tp + fp, 37525, "Evaluated changed pixels must equal 37,525")
        self.assertEqual(tp + fn, 35362, "Reference water pixels must equal 35,362")
        self.assertEqual(tn + fp, 52715, "Reference dry pixels must equal 52,715")
        self.assertEqual(tp + fp + fn + tn, 88077, "Total evaluated pixels must equal 88,077")

        # Reconcile Domain A (45,788) vs Domain C (37,525)
        changed_in_excluded = int(np.sum((self.delta_vv < -5.0) & self.annotator_excluded_mask))
        self.assertEqual(changed_in_excluded, 8263, "Changed pixels in annotator-excluded region must equal 8,263")
        self.assertEqual((tp + fp) + changed_in_excluded, 45788, "Evaluated changed (37,525) + Excluded changed (8,263) must equal 45,788")

    def test_04_mathematical_metric_derivations(self):
        """Verify mathematical derivation of precision, recall, IoU, and F1."""
        tp = 29249
        fp = 8276
        fn = 6113
        tn = 44439

        expected_prec = tp / (tp + fp)
        expected_rec = tp / (tp + fn)
        expected_iou = tp / (tp + fp + fn)
        expected_f1 = 2 * expected_prec * expected_rec / (expected_prec + expected_rec)

        metrics = self.metrics_data["exploratory_evaluation_metrics"]["primary_baseline_threshold_minus_5db"]

        self.assertAlmostEqual(metrics["precision"], expected_prec, places=6)
        self.assertAlmostEqual(metrics["recall"], expected_rec, places=6)
        self.assertAlmostEqual(metrics["iou"], expected_iou, places=6)
        self.assertAlmostEqual(metrics["f1_score"], expected_f1, places=6)
        
        self.assertAlmostEqual(metrics["precision"], 0.7794537, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.8271308, places=4)
        self.assertAlmostEqual(metrics["iou"], 0.6702644, places=4)
        self.assertAlmostEqual(metrics["f1_score"], 0.8025848, places=4)

    def test_05_geodesic_area_reconciliation(self):
        """Verify per-pixel geodesic area calculations and domain reconciliation."""
        domain_a = self.metrics_data["domain_decomposition"]["domain_A_full_sar_swath"]
        domain_c = self.metrics_data["domain_decomposition"]["domain_C_evaluation_domain"]
        region_d = self.metrics_data["domain_decomposition"]["region_D_annotator_excluded"]

        # Pixel counts
        self.assertEqual(domain_a["pixel_count"], 101623)
        self.assertEqual(domain_c["pixel_count"], 88077)
        self.assertEqual(region_d["pixel_count"], 13546)

        # Changed areas
        self.assertAlmostEqual(domain_a["changed_area_ha"], 441.20, places=1)
        self.assertAlmostEqual(domain_c["evaluated_changed_area_ha"], 361.58, places=1)
        self.assertAlmostEqual(region_d["changed_area_ha"], 79.62, places=1)

        # Reconciliation: Domain C + Region D == Domain A
        reconciled_sum = domain_c["evaluated_changed_area_ha"] + region_d["changed_area_ha"]
        self.assertAlmostEqual(reconciled_sum, domain_a["changed_area_ha"], places=2)

    def test_06_cross_artifact_synchronization(self):
        """Verify that JSON, CSV, and Markdown report figures are strictly identical."""
        # Find row for -5.0 in CSV
        row_minus_5 = next((r for r in self.csv_rows if r["threshold_dB"] == "-5.0"), None)
        self.assertIsNotNone(row_minus_5, "CSV must contain row for -5.0 dB")
        
        self.assertEqual(int(row_minus_5["tp"]), 29249)
        self.assertEqual(int(row_minus_5["fp"]), 8276)
        self.assertEqual(int(row_minus_5["fn"]), 6113)
        self.assertEqual(int(row_minus_5["tn"]), 44439)
        self.assertEqual(int(row_minus_5["changed_pixels_eval_domain"]), 37525)

        # Check Report mentions both reconciled numbers
        self.assertIn("37,525", self.report_text, "Report must explain 37,525 evaluated changed pixels")
        self.assertIn("45,788", self.report_text, "Report must explain 45,788 full swath changed pixels")
        self.assertIn("13,546", self.report_text, "Report must explain 13,546 annotator excluded pixels")
        self.assertIn("8,263", self.report_text, "Report must explain 8,263 excluded changed pixels")
        self.assertIn("441.20", self.report_text, "Report must report 441.20 ha for swath changed area")
        self.assertIn("361.58", self.report_text, "Report must report 361.58 ha for evaluated changed area")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFloodChangeDetectionIntegrity)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

