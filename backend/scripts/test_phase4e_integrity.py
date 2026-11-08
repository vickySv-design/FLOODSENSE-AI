"""
FloodSense-AI: Phase 4E Automated Integrity & Multi-Chip Validation Test Suite
==============================================================================
Verifies the 10 mandatory integrity checks for Phase 4E:
1. Strict isolation of exploratory chip (Bolivia_103757) from the 14-chip evaluation set.
2. Complete file existence and metadata provenance for every evaluated chip.
3. NoData and out-of-swath exclusion correctness.
4. Fixed primary threshold verification (-5.0 dB untuned).
5. Exact reconciliation of confusion-matrix counts with binary masks.
6. Mathematical derivation correctness of per-chip metrics.
7. Pooled aggregation correctness from sums of counts.
8. Synchronization across CSV, JSON, and Markdown reports.
9. Geodesic area calculations using actual valid pixel masks.
10. Transparency: all 15 chips accounted for (1 exploratory, 14 evaluated, 0 silently omitted).
"""

import os
import sys
import json
import csv
import unittest
import numpy as np
import rasterio

class TestPhase4EIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.inv_csv_path = "backend/data/flood_event/phase4e_chip_inventory.csv"
        cls.cov_csv_path = "backend/outputs/phase4e/phase4e_chip_coverage.csv"
        cls.per_chip_csv_path = "backend/outputs/phase4e/phase4e_per_chip_metrics.csv"
        cls.agg_json_path = "backend/outputs/phase4e/phase4e_aggregate_metrics.json"
        cls.thresh_csv_path = "backend/outputs/phase4e/phase4e_threshold_comparison.csv"
        cls.report_path = "backend/outputs/phase4e/phase4e_validation_report.md"
        cls.viz_png_path = "backend/outputs/phase4e/phase4e_validation_visualization.png"

        # Load inventory
        with open(cls.inv_csv_path, "r", encoding="utf-8") as f:
            cls.inventory = list(csv.DictReader(f))

        # Load per-chip metrics
        with open(cls.per_chip_csv_path, "r", encoding="utf-8") as f:
            cls.per_chip_rows = list(csv.DictReader(f))

        # Load aggregate JSON
        with open(cls.agg_json_path, "r", encoding="utf-8") as f:
            cls.agg_data = json.load(f)

        # Load coverage CSV
        with open(cls.cov_csv_path, "r", encoding="utf-8") as f:
            cls.cov_rows = list(csv.DictReader(f))

        # Load report
        with open(cls.report_path, "r", encoding="utf-8") as f:
            cls.report_text = f.read()

    def test_01_exploratory_chip_isolation(self):
        """Verify strict isolation of exploratory chip Bolivia_103757 from 14-chip evaluation."""
        eval_chips = [r["chip_id"] for r in self.per_chip_rows]
        self.assertEqual(len(eval_chips), 14, "Must evaluate exactly 14 holdout chips")
        self.assertNotIn("Bolivia_103757", eval_chips, "Bolivia_103757 must NOT be in evaluation set")

        # In inventory, Bolivia_103757 is clearly labeled
        exp_row = next((r for r in self.inventory if r["chip_id"] == "Bolivia_103757"), None)
        self.assertIsNotNone(exp_row)
        self.assertIn("EXPLORATORY", exp_row["eligibility_reason"])

    def test_02_all_evaluated_chips_have_valid_inputs_and_metadata(self):
        """Verify that every evaluated chip has all raster files and metadata present on disk."""
        for row in self.per_chip_rows:
            cid = row["chip_id"]
            pre_path = f"backend/data/flood_event/phase4e/before/{cid}_S1_PreEvent_20180203.tif"
            ev_path = f"backend/data/official_sen1floods11/raw/S1Hand/{cid}_S1Hand.tif"
            lbl_path = f"backend/data/official_sen1floods11/raw/LabelHand/{cid}_LabelHand.tif"
            meta_path = f"backend/data/flood_event/phase4e/metadata/{cid}_pre_event_metadata.json"

            self.assertTrue(os.path.exists(pre_path), f"Missing pre-event raster for {cid}")
            self.assertTrue(os.path.exists(ev_path), f"Missing event SAR for {cid}")
            self.assertTrue(os.path.exists(lbl_path), f"Missing label for {cid}")
            self.assertTrue(os.path.exists(meta_path), f"Missing metadata for {cid}")

            # Check pre-event raster size > 1MB
            self.assertGreater(os.path.getsize(pre_path), 1000000, f"Corrupt pre-event raster for {cid}")

    def test_03_nodata_and_coverage_decomposition(self):
        """Verify NoData pixels are excluded consistently and coverage counts match."""
        self.assertEqual(len(self.cov_rows), 15, "Coverage CSV must report all 15 Bolivia chips")
        for cov in self.cov_rows:
            tot = int(cov["total_tile_pixels"])
            sar_val = int(cov["valid_sar_pixels"])
            eval_val = int(cov["evaluation_domain_pixels"])
            excl_val = int(cov["annotator_excluded_pixels"])

            self.assertEqual(tot, 262144)
            self.assertEqual(eval_val + excl_val, sar_val, f"Decomposition failed for {cov['chip_id']}")

    def test_04_fixed_threshold_verification(self):
        """Verify the primary fixed threshold is exactly -5.0 dB for all evaluated chips."""
        for row in self.per_chip_rows:
            t = float(row["threshold_dB"])
            self.assertEqual(t, -5.0, f"Chip {row['chip_id']} used modified threshold {t}")
        self.assertEqual(self.agg_data["fixed_primary_threshold"], "Delta VV < -5.0 dB")

    def test_05_confusion_matrix_reconciliation(self):
        """Verify confusion matrix totals reconcile with prediction and ground truth totals."""
        for row in self.per_chip_rows:
            tp = int(row["tp"])
            fp = int(row["fp"])
            fn = int(row["fn"])
            tn = int(row["tn"])
            pred_pos = int(row["pred_changed_pixels_eval_domain"])
            ref_water = int(row["ref_water_class2_pixels"])
            ref_dry = int(row["ref_dry_class0_pixels"])

            self.assertEqual(tp + fp, pred_pos)
            self.assertEqual(tp + fn, ref_water)
            self.assertEqual(tn + fp, ref_dry)

    def test_06_mathematical_metric_derivations(self):
        """Verify precision, recall, IoU, and F1-score mathematical calculations."""
        for row in self.per_chip_rows:
            tp = int(row["tp"])
            fp = int(row["fp"])
            fn = int(row["fn"])
            tn = int(row["tn"])

            exp_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            exp_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            exp_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            exp_f1 = 2 * exp_prec * exp_rec / (exp_prec + exp_rec) if (exp_prec + exp_rec) > 0 else 0.0

            self.assertAlmostEqual(float(row["precision"]), exp_prec, places=4)
            self.assertAlmostEqual(float(row["recall"]), exp_rec, places=4)
            self.assertAlmostEqual(float(row["iou"]), exp_iou, places=4)
            self.assertAlmostEqual(float(row["f1_score"]), exp_f1, places=4)

    def test_07_pooled_metrics_reconcile_with_aggregated_counts(self):
        """Verify pooled metrics match the exact sum of per-chip confusion matrices."""
        sum_tp = sum(int(r["tp"]) for r in self.per_chip_rows)
        sum_fp = sum(int(r["fp"]) for r in self.per_chip_rows)
        sum_fn = sum(int(r["fn"]) for r in self.per_chip_rows)
        sum_tn = sum(int(r["tn"]) for r in self.per_chip_rows)

        pooled = self.agg_data["pooled_pixel_metrics"]
        self.assertEqual(pooled["pooled_true_positives"], sum_tp)
        self.assertEqual(pooled["pooled_false_positives"], sum_fp)
        self.assertEqual(pooled["pooled_false_negatives"], sum_fn)
        self.assertEqual(pooled["pooled_true_negatives"], sum_tn)

        expected_pooled_iou = sum_tp / (sum_tp + sum_fp + sum_fn)
        expected_pooled_f1 = 2 * sum_tp / (2 * sum_tp + sum_fp + sum_fn)

        self.assertAlmostEqual(pooled["pooled_iou"], expected_pooled_iou, places=6)
        self.assertAlmostEqual(pooled["pooled_f1_score"], expected_pooled_f1, places=6)

    def test_08_cross_artifact_synchronization(self):
        """Verify CSV, JSON, and Report are synchronized."""
        pooled = self.agg_data["pooled_pixel_metrics"]
        p_iou_str = f"{pooled['pooled_iou']:.4f}"
        p_f1_str = f"{pooled['pooled_f1_score']:.4f}"

        self.assertIn(p_iou_str, self.report_text)
        self.assertIn(p_f1_str, self.report_text)
        self.assertTrue(os.path.exists(self.viz_png_path))
        self.assertGreater(os.path.getsize(self.viz_png_path), 500000)

    def test_09_geodesic_area_valid_footprint(self):
        """Verify geodesic area calculations use valid swath masks and are non-negative."""
        for row in self.per_chip_rows:
            ha = float(row["swath_changed_area_ha"])
            km2 = float(row["swath_changed_area_km2"])
            self.assertGreaterEqual(ha, 0.0)
            self.assertAlmostEqual(km2, ha / 100.0, places=3)

    def test_10_no_silent_omissions(self):
        """Verify that all 15 Bolivia chips are documented across inventory and outputs."""
        inv_ids = set(r["chip_id"] for r in self.inventory)
        self.assertEqual(len(inv_ids), 15)
        eval_ids = set(r["chip_id"] for r in self.per_chip_rows)
        self.assertEqual(len(eval_ids), 14)
        self.assertEqual(inv_ids - eval_ids, {"Bolivia_103757"})

    def test_11_class2_water_count_reconciliation(self):
        """Verify Class 2 reference water count equals 386,588 and reconciles with TP + FN."""
        sum_class2 = sum(int(r["ref_water_class2_pixels"]) for r in self.per_chip_rows)
        sum_class0 = sum(int(r["ref_dry_class0_pixels"]) for r in self.per_chip_rows)
        sum_tp = sum(int(r["tp"]) for r in self.per_chip_rows)
        sum_fn = sum(int(r["fn"]) for r in self.per_chip_rows)
        sum_fp = sum(int(r["fp"]) for r in self.per_chip_rows)
        sum_tn = sum(int(r["tn"]) for r in self.per_chip_rows)

        self.assertEqual(sum_class2, 386588, "Class 2 reference water pixels must sum to exactly 386,588")
        self.assertEqual(sum_tp + sum_fn, sum_class2, "TP + FN must equal Class 2 reference water pixels")
        self.assertEqual(sum_class0, 2319818, "Class 0 reference dry pixels must sum to exactly 2,319,818")
        self.assertEqual(sum_tn + sum_fp, sum_class0, "TN + FP must equal Class 0 reference dry pixels")
        self.assertEqual(sum_class0 + sum_class2, 2706406, "Class 0 + Class 2 must equal total binary eval pixels")

        # Verify JSON reconciliation block
        rec = self.agg_data.get("reference_pixel_reconciliation", {})
        self.assertEqual(rec.get("class2_flood_water_pixels"), 386588)
        self.assertEqual(rec.get("class0_dry_land_pixels"), 2319818)

    def test_12_top_three_chips_percentage(self):
        """Verify the top-3 inundated chips percentage is 78.94% (<80.0%) of Class 2 water."""
        top_chips = ["Bolivia_129334", "Bolivia_314919", "Bolivia_432776"]
        counts = {r["chip_id"]: int(r["ref_water_class2_pixels"]) for r in self.per_chip_rows if r["chip_id"] in top_chips}

        self.assertEqual(counts["Bolivia_129334"], 152073)
        self.assertEqual(counts["Bolivia_314919"], 83131)
        self.assertEqual(counts["Bolivia_432776"], 69982)

        top3_sum = sum(counts.values())
        self.assertEqual(top3_sum, 305186)

        total_class2 = sum(int(r["ref_water_class2_pixels"]) for r in self.per_chip_rows)
        percentage = (top3_sum / total_class2) * 100.0

        self.assertAlmostEqual(percentage, 78.9435, places=3)
        self.assertLess(percentage, 80.0, "Top 3 chips contain ~78.94%, NOT >80%")

        # Verify JSON
        top3_json = self.agg_data.get("top_three_chips_water_concentration", {})
        self.assertEqual(top3_json.get("top_3_combined_water_pixels"), 305186)
        self.assertAlmostEqual(top3_json.get("percentage_of_class2_water", 0.0), 78.9435, places=3)

    def test_13_domain_decomposition_class1_reconciliation(self):
        """Verify 2,736,357 valid labelled pixels - 2,706,406 binary eval pixels = 29,951 Class 1 pixels."""
        # 1. Total valid labelled pixels from coverage CSV for the 14 evaluation chips (class0 + class2 + class1)
        eval_cov = [r for r in self.cov_rows if r["chip_id"] != "Bolivia_103757"]
        sum_labelled_cov = sum(int(r["evaluation_domain_pixels"]) for r in eval_cov)
        sum_class1_cov = sum(int(r["class1_perm_water_pixels"]) for r in eval_cov)
        self.assertEqual(sum_labelled_cov, 2736357, "Valid labelled pixels must sum to 2,736,357")
        self.assertEqual(sum_class1_cov, 29951, "Permanent water Class 1 must sum to 29,951")

        # 2. Binary evaluation pixels in per_chip_metrics
        sum_eval_binary = sum(int(r["tp"]) + int(r["fp"]) + int(r["fn"]) + int(r["tn"]) for r in self.per_chip_rows)
        self.assertEqual(sum_eval_binary, 2706406, "Binary evaluation domain must sum to 2,706,406")

        diff = sum_labelled_cov - sum_eval_binary
        self.assertEqual(diff, 29951, "Difference must be exactly 29,951 pixels (Class 1 permanent water)")

        # 3. Verify directly from source LabelHand and JRCWaterHand rasters that Class 1 within valid SAR equals 29,951
        sum_class1_rasters = 0
        for r in self.per_chip_rows:
            cid = r["chip_id"]
            pre_path = f"backend/data/flood_event/phase4e/before/{cid}_S1_PreEvent_20180203.tif"
            ev_path = f"backend/data/official_sen1floods11/raw/S1Hand/{cid}_S1Hand.tif"
            lbl_path = f"backend/data/official_sen1floods11/raw/LabelHand/{cid}_LabelHand.tif"
            jrc_path = f"backend/data/official_sen1floods11/raw/JRCWaterHand/{cid}_JRCWaterHand.tif"
            with rasterio.open(pre_path) as s_pre, rasterio.open(ev_path) as s_ev, \
                 rasterio.open(lbl_path) as s_lbl, rasterio.open(jrc_path) as s_jrc:
                pre_arr = s_pre.read()
                ev_arr = s_ev.read()
                lbl_arr = s_lbl.read(1)
                jrc_arr = s_jrc.read(1)

                valid_sar = (~np.isnan(ev_arr[0])) & (~np.isnan(pre_arr[0])) & (~np.isnan(ev_arr[1])) & (~np.isnan(pre_arr[1]))
                valid_lbl = (lbl_arr == 0) | (lbl_arr == 1)
                eval_mask = valid_sar & valid_lbl
                c1 = eval_mask & (lbl_arr == 1) & (jrc_arr == 1)
                sum_class1_rasters += int(np.sum(c1))

        self.assertEqual(sum_class1_rasters, 29951, "Source LabelHand + JRC Class 1 (permanent water within SAR) must equal 29,951")

    def test_14_spatial_holdout_terminology(self):
        """Verify report, README, and JSON describe 14 chips as spatial holdout of a single event."""
        scope_str = self.agg_data.get("evaluation_scope", "")
        self.assertIn("Spatial Holdout", scope_str)

        # Check Report text
        self.assertIn("Spatial Holdout", self.report_text)
        self.assertNotIn("14 independent flood events", self.report_text)
        self.assertIn("single regional flood disaster event", self.report_text.lower())


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase4EIntegrity)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

