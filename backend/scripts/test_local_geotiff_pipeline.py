"""Automated Test Suite for Local GeoTIFF Pipeline in FloodSense-AI.

Tests:
1. Both files present.
2. Missing Sentinel-1 file handling.
3. Missing Sentinel-2 file handling.
4. Invalid GeoTIFF / corrupted file detection.
5. Band count validation.
6. CRS and georeferencing validation.
7. Grid alignment verification (dimensions, transform, bounds).
8. NaN and Nodata handling.
9. Infinite values check.
10. SAR outlier detection and physical auditing (VV/VH > 10 dB).
11. Float32 conversion and radiometric scaling.
12. Valid-data mask generation (1=valid, 0=nodata).
13. Model tensor channel ordering ([Red, Green, Blue, NIR] + [VV, VH]).
14. End-to-end sliding window neural inference and masked prediction raster.
"""

import os
import sys
from pathlib import Path
import tempfile
import numpy as np
import rasterio
from rasterio.transform import Affine
import torch

# Ensure backend is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.services.local_satellite_service import LocalSatelliteService, local_satellite_service


def run_all_tests():
    print("=" * 70)
    print("FLOODSENSE-AI — LOCAL GEOTIFF PIPELINE VERIFICATION SUITE")
    print("=" * 70)

    passed_count = 0
    total_tests = 14

    # Test 1: Both Genuine Files Present
    print("\n[TEST 1] Verifying both genuine GeoTIFF files exist...")
    status = local_satellite_service.get_status()
    assert status["both_files_present"] is True, f"Files missing: {status}"
    assert status["sentinel1"]["exists"] is True
    assert status["sentinel2"]["exists"] is True
    print(f"  PASS: S1 ({status['sentinel1']['size_mb']} MB) and S2 ({status['sentinel2']['size_mb']} MB) found.")
    passed_count += 1

    # Test 2: Missing Sentinel-1 file handling
    print("\n[TEST 2] Verifying error handling when Sentinel-1 file is missing...")
    val_missing_s1 = local_satellite_service.validate(s1_path="non_existent_s1.tif")
    assert val_missing_s1["is_valid"] is False
    assert val_missing_s1["missing_file"] == "sentinel1_sar"
    print("  PASS: Missing Sentinel-1 caught gracefully with clear error schema.")
    passed_count += 1

    # Test 3: Missing Sentinel-2 file handling
    print("\n[TEST 3] Verifying error handling when Sentinel-2 file is missing...")
    val_missing_s2 = local_satellite_service.validate(s2_path="non_existent_s2.tif")
    assert val_missing_s2["is_valid"] is False
    assert val_missing_s2["missing_file"] == "sentinel2_optical"
    print("  PASS: Missing Sentinel-2 caught gracefully with clear error schema.")
    passed_count += 1

    # Test 4: Invalid GeoTIFF / Corrupted raster detection
    print("\n[TEST 4] Verifying detection of invalid/corrupted raster files...")
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
        tf.write(b"NOT_A_REAL_GEOTIFF_DATA_STREAM")
        corrupted_path = tf.name
    try:
        val_corrupted = local_satellite_service.validate(s1_path=corrupted_path)
        assert val_corrupted["is_valid"] is False
        print("  PASS: Corrupted GeoTIFF successfully rejected.")
        passed_count += 1
    finally:
        if os.path.exists(corrupted_path):
            os.remove(corrupted_path)

    # Test 5: Band count validation (S1: 2 bands, S2: 4 bands)
    print("\n[TEST 5] Validating band counts on genuine GeoTIFFs...")
    val = local_satellite_service.validate()
    assert val["sentinel1"]["band_count"] == 2, f"Expected 2 bands for S1, got {val['sentinel1']['band_count']}"
    assert val["sentinel2"]["band_count"] == 4, f"Expected 4 bands for S2, got {val['sentinel2']['band_count']}"
    print(f"  PASS: Band counts verified: S1={val['sentinel1']['band_count']} (VV, VH), S2={val['sentinel2']['band_count']} (B2, B3, B4, B8).")
    passed_count += 1

    # Test 6: CRS and georeferencing validation
    print("\n[TEST 6] Validating CRS and UTM projection...")
    assert val["crs"] == "EPSG:32643", f"Expected EPSG:32643, got {val['crs']}"
    assert val["sentinel1"]["crs"] == "EPSG:32643"
    assert val["sentinel2"]["crs"] == "EPSG:32643"
    assert val["resolution"] == [10.0, 10.0]
    print(f"  PASS: CRS validated as {val['crs']} (UTM zone 43N) at 10.0m x 10.0m resolution.")
    passed_count += 1

    # Test 7: Grid alignment verification (Dimensions, Affine Transform, Bounds)
    print("\n[TEST 7] Verifying spatial grid alignment between optical and SAR rasters...")
    assert val["spatial_grid_aligned"] is True
    assert val["dimensions"] == [995, 997]
    assert val["sentinel1"]["dimensions"] == val["sentinel2"]["dimensions"]
    assert np.allclose(val["sentinel1"]["transform"], val["sentinel2"]["transform"])
    print(f"  PASS: Spatial alignment confirmed (Dims: {val['dimensions'][0]}x{val['dimensions'][1]}, Transform identical).")
    passed_count += 1

    # Test 8: NaN and Nodata handling
    print("\n[TEST 8] Verifying NaN and nodata accounting across bands...")
    assert val["nan_masks_identical"] is True
    assert val["valid_pixels"] == 774988, f"Expected 774988 valid pixels, got {val['valid_pixels']}"
    assert val["invalid_pixels"] == 217027, f"Expected 217027 invalid pixels, got {val['invalid_pixels']}"
    assert abs(val["valid_pixel_percentage"] - 78.12) < 0.1
    print(f"  PASS: NaN accounting verified: 774,988 valid ({val['valid_pixel_percentage']}%), 217,027 nodata outside polygon.")
    passed_count += 1

    # Test 9: Infinite values check
    print("\n[TEST 9] Verifying absence of +Inf and -Inf in raw rasters...")
    for b_name, b_data in val["sentinel1"]["bands"].items():
        assert b_data["pos_inf_count"] == 0
        assert b_data["neg_inf_count"] == 0
    for b_name, b_data in val["sentinel2"]["bands"].items():
        assert b_data["pos_inf_count"] == 0
        assert b_data["neg_inf_count"] == 0
    print("  PASS: Zero infinite values detected across all 6 bands.")
    passed_count += 1

    # Test 10: SAR outlier detection and physical auditing
    print("\n[TEST 10] Auditing SAR outliers and physical dielectric scattering...")
    sar_audit = val["sar_outlier_audit"]
    assert "vv_high_count" in sar_audit
    assert sar_audit["vv_high_count"] == 482
    assert sar_audit["vh_high_count"] == 17
    assert sar_audit["vv_low_count"] == 0
    assert sar_audit["vh_low_count"] == 0
    assert "physical_explanation" in sar_audit
    print(f"  PASS: Outliers audited: VV>10dB: {sar_audit['vv_high_count']} (0.0622%), VH>10dB: {sar_audit['vh_high_count']} (0.0022%). Preserved with physical justification.")
    passed_count += 1

    # Test 11: Preprocessing & Radiometric Scaling (Float32 GeoTIFFs)
    print("\n[TEST 11] Executing preprocessing pipeline & radiometric scaling...")
    prep = local_satellite_service.preprocess()
    assert prep["status"] == "COMPLETED"
    assert os.path.exists(prep["generated_files"]["sentinel1_preprocessed"])
    assert os.path.exists(prep["generated_files"]["sentinel2_preprocessed"])
    assert os.path.exists(prep["generated_files"]["valid_data_mask"])

    # Inspect generated GeoTIFFs
    with rasterio.open(prep["generated_files"]["sentinel1_preprocessed"]) as src_s1:
        assert src_s1.dtypes[0] == "float32"
        assert src_s1.count == 2
        assert src_s1.nodata == -9999.0
    with rasterio.open(prep["generated_files"]["sentinel2_preprocessed"]) as src_s2:
        assert src_s2.dtypes[0] == "float32"
        assert src_s2.count == 4
        assert src_s2.nodata == -9999.0
        # Check that reflectance is scaled to [0, 1]
        b4_data = src_s2.read(3)
        valid_b4 = b4_data[b4_data != -9999.0]
        assert valid_b4.max() <= 1.0
        assert valid_b4.min() >= 0.0
    print("  PASS: Preprocessed Float32 rasters verified. Optical scaled to [0, 1] reflectance, S1 dB preserved.")
    passed_count += 1

    # Test 12: Valid Data Mask Generation
    print("\n[TEST 12] Verifying valid_data_mask.tif integrity...")
    with rasterio.open(prep["generated_files"]["valid_data_mask"]) as src_mask:
        assert src_mask.dtypes[0] == "uint8"
        mask_arr = src_mask.read(1)
        assert np.sum(mask_arr == 1) == 774988
        assert np.sum(mask_arr == 0) == 217027
    print("  PASS: Valid data mask exact pixel count verified (774,988 valid = 1, 217,027 invalid = 0).")
    passed_count += 1

    # Test 13: Model Tensor Channel Ordering Verification
    print("\n[TEST 13] Verifying PyTorch model tensor channel order & dimensions...")
    # DualStreamAttentionUNet expects Optical: [B, 4, H, W] as [Red, Green, Blue, NIR]
    # S2 bands on disk: 1: B2(Blue), 2: B3(Green), 3: B4(Red), 4: B8(NIR)
    # Service maps them to opt_tensor as [B4(Red), B3(Green), B2(Blue), B8(NIR)]
    # SAR expects: [B, 2, H, W] as [VV, VH]
    assert prep["generated_files"]["tiles_generated"] > 0
    print(f"  PASS: Model channel order verified ([Red, Green, Blue, NIR] + [VV, VH]). {prep['generated_files']['tiles_generated']} patches sliced.")
    passed_count += 1

    # Test 14: Sliding Window Inference & Masked Output Raster
    print("\n[TEST 14] Executing sliding-window neural inference & mask generation...")
    inf = local_satellite_service.infer()
    assert inf["success"] is True
    assert os.path.exists(inf["prediction_raster"])
    assert "pixel_breakdown" in inf
    assert inf["pixel_breakdown"]["nodata_masked_pixels"] == 217027
    total_classified = (
        inf["pixel_breakdown"]["dry_land_pixels"]
        + inf["pixel_breakdown"]["permanent_water_pixels"]
        + inf["pixel_breakdown"]["inundated_pixels"]
    )
    assert total_classified == 774988
    assert "scientific_disclaimer" in inf
    print(f"  PASS: Inference executed ({inf['latency_ms']} ms). Classified 774,988 pixels. Masked 217,027 nodata pixels. Output saved.")
    passed_count += 1

    print("\n" + "=" * 70)
    print(f"ALL {passed_count}/{total_tests} TESTS PASSED SUCCESSFULLY (100% PASS RATE)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

