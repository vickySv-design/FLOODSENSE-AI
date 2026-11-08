"""
Comprehensive Dual-Provider Test Suite for FloodSense-AI
Verifies:
1. Copernicus Data Space Ecosystem (CDSE) Adapter
2. Google Earth Engine (GEE) Adapter
3. RasterValidator Integrity Engine
4. DataProvenance Persistence
5. Dual-Stream Attention U-Net Inference on Acquired Rasters
6. Offline and Hybrid Fallback Handlers
"""
import os
import sys
import json
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import PRESETS_DIR, SATELLITE_CACHE_DIR, METADATA_DIR, DEVICE_NAME
from app.services.data_providers.copernicus_provider import CopernicusDataSpaceProvider
from app.services.data_providers.gee_provider import GoogleEarthEngineProvider
from app.services.data_acquisition_manager import data_acquisition_manager
from app.services.raster_validator import raster_validator
from app.services.data_provenance import data_provenance_manager
from app.services.preprocessor import preprocess_multimodal_pair
from app.services.inference_engine import inference_engine

def run_tests():
    print("=" * 80)
    print("FLOODSENSE-AI: DUAL-PROVIDER SATELLITE INTEGRATION TEST SUITE")
    print(f"Hardware Engine: {DEVICE_NAME}")
    print("=" * 80)

    aoi_bbox = [13.01, 80.22, 13.06, 80.27] # Chennai Coastal Corridor

    # -------------------------------------------------------------------------
    # TEST 1: CDSE Connection & Diagnostics
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Testing Copernicus Data Space Connection & Diagnostics...")
    cop = CopernicusDataSpaceProvider()
    cop_status = cop.get_provider_status()
    print(f"  CDSE Provider Name: {cop_status['name']}")
    print(f"  CDSE Status: {cop_status['status']}")
    print(f"  CDSE Connected: {cop_status['connected']}")
    assert "status" in cop_status
    print(">> PASSED: CDSE diagnostic status checked without errors.")

    # -------------------------------------------------------------------------
    # TEST 2: CDSE Product Search (Sentinel-2 & Sentinel-1)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Testing Copernicus CDSE Scene Search...")
    s2_scenes = cop.search_products(tuple(aoi_bbox), "2025-11-01", "2025-11-30", platform="SENTINEL-2")
    s1_scenes = cop.search_products(tuple(aoi_bbox), "2025-12-01", "2025-12-15", platform="SENTINEL-1")
    print(f"  S2 Scenes Found: {len(s2_scenes)} (First: {s2_scenes[0].scene_id})")
    print(f"  S1 Scenes Found: {len(s1_scenes)} (First: {s1_scenes[0].scene_id})")
    assert len(s2_scenes) > 0
    assert len(s1_scenes) > 0
    assert s2_scenes[0].available_bands == ["B02", "B03", "B04", "B08"]
    assert s1_scenes[0].polarizations == ["VV", "VH"]
    print(">> PASSED: CDSE product search returned standardized metadata objects.")

    # -------------------------------------------------------------------------
    # TEST 3: CDSE Product Download & Genuine GeoTIFF Raster Extraction
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Testing CDSE GeoTIFF Download & Extraction...")
    downloaded_s2 = cop.download_product(s2_scenes[0], ["B02", "B03", "B04", "B08"])
    downloaded_s1 = cop.download_product(s1_scenes[0], ["VV", "VH"])
    print(f"  S2 Band Files Created: {list(downloaded_s2.keys())}")
    print(f"  S1 Band Files Created: {list(downloaded_s1.keys())}")
    assert len(downloaded_s2) == 4
    assert len(downloaded_s1) == 2
    for b, p in {**downloaded_s2, **downloaded_s1}.items():
        assert os.path.exists(p)
        assert os.path.getsize(p) > 1000
    print(">> PASSED: Genuine raster files written to local cache (non-empty, > 1000 bytes).")

    # -------------------------------------------------------------------------
    # TEST 4: Physical Raster Validation with RasterValidator
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Validating Physical Rasters using RasterValidator...")
    b04_val = raster_validator.validate_raster_file(downloaded_s2["B04"])
    vv_val = raster_validator.validate_raster_file(downloaded_s1["VV"])
    print(f"  B04 Validation: Format={b04_val['format']}, Integrity={b04_val['integrity_status']}, CRS={b04_val.get('crs')}")
    print(f"  VV Validation: Format={vv_val['format']}, Integrity={vv_val['integrity_status']}, Dims={vv_val.get('dimensions')}")
    assert b04_val["is_valid"] is True
    assert vv_val["is_valid"] is True
    assert b04_val["integrity_status"] == "VERIFIED"
    assert vv_val["integrity_status"] == "VERIFIED"
    print(">> PASSED: RasterValidator confirmed GeoTIFF CRS, affine transform, and non-zero pixels.")

    # -------------------------------------------------------------------------
    # TEST 5: GEE Connection & Diagnostics
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Testing Google Earth Engine Connection & Diagnostics...")
    gee = GoogleEarthEngineProvider()
    gee_status = gee.get_provider_status()
    print(f"  GEE Provider Name: {gee_status['name']}")
    print(f"  GEE Status: {gee_status['status']}")
    print(f"  GEE Connected: {gee_status['connected']}")
    assert "status" in gee_status
    print(">> PASSED: GEE diagnostic status verified without throwing unhandled exceptions.")

    # -------------------------------------------------------------------------
    # TEST 6: GEE Product Search & Scene Metadata
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Testing GEE Product Search (Sentinel-2 & Sentinel-1)...")
    gee_s2 = gee.search_products(tuple(aoi_bbox), "2025-11-01", "2025-11-30", platform="SENTINEL-2")
    gee_s1 = gee.search_products(tuple(aoi_bbox), "2025-12-01", "2025-12-15", platform="SENTINEL-1")
    print(f"  GEE S2 Scenes: {len(gee_s2)} (First: {gee_s2[0].scene_id})")
    print(f"  GEE S1 Scenes: {len(gee_s1)} (First: {gee_s1[0].scene_id})")
    assert len(gee_s2) > 0
    assert len(gee_s1) > 0
    print(">> PASSED: GEE search returns standardized SatelliteSceneMetadata.")

    # -------------------------------------------------------------------------
    # TEST 7: GEE GeoTIFF Extraction & Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Testing GEE GeoTIFF Download & Extraction...")
    gee_s2_paths = gee.download_product(gee_s2[0], ["B02", "B03", "B04", "B08"])
    gee_s1_paths = gee.download_product(gee_s1[0], ["VV", "VH"])
    print(f"  GEE S2 Downloaded Bands: {list(gee_s2_paths.keys())}")
    print(f"  GEE S1 Downloaded Bands: {list(gee_s1_paths.keys())}")
    assert len(gee_s2_paths) == 4
    assert len(gee_s1_paths) == 2
    gee_val = raster_validator.validate_raster_file(gee_s2_paths["B02"])
    assert gee_val["is_valid"] is True
    print(">> PASSED: GEE adapter produced genuine validated GeoTIFF raster arrays.")

    # -------------------------------------------------------------------------
    # TEST 8: Full Data Acquisition Manager Multi-Modal Pair Pipeline
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Testing Central Data Acquisition Manager acquire_multimodal_pair()...")
    pair_res = data_acquisition_manager.acquire_multimodal_pair(
        bbox=aoi_bbox,
        pre_date="2025-11-10",
        post_date="2025-12-05",
        provider="copernicus",
        aoi_name="Chennai Test Sector",
    )
    print(f"  Acquisition ID: {pair_res['analysis_id']}")
    print(f"  Status: {pair_res['status']}")
    print(f"  Provenance Tag: {pair_res['provenance']['provenance_tag']}")
    assert pair_res["status"] == "ACQUIRED_AND_VALIDATED"
    assert "optical_band_paths" in pair_res
    assert "sar_band_paths" in pair_res
    print(">> PASSED: Multi-modal acquisition pipeline completed end-to-end.")

    # -------------------------------------------------------------------------
    # TEST 9: Data Provenance Manifest Persistence
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Verifying Immutable Data Provenance Record...")
    prov = data_provenance_manager.get_provenance(pair_res["analysis_id"])
    assert prov is not None
    print(f"  Persisted Manifest Path: {prov.get('saved_path')}")
    print(f"  Optical Sensor: {prov['satellite_acquisitions']['pre_disaster_optical']['platform']}")
    print(f"  SAR Sensor: {prov['satellite_acquisitions']['post_disaster_sar']['platform']}")
    assert prov["satellite_acquisitions"]["pre_disaster_optical"]["platform"] == "SENTINEL-2 MSI"
    assert prov["satellite_acquisitions"]["post_disaster_sar"]["platform"] == "SENTINEL-1 C-SAR"
    print(">> PASSED: Provenance manifest correctly persisted and retrieved from disk.")

    # -------------------------------------------------------------------------
    # TEST 10: Deep Learning Inference on Acquired Rasters
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Running Dual-Stream Attention U-Net on Acquired Rasters...")
    # Read raster bands using rasterio
    import rasterio
    with rasterio.open(downloaded_s2["B04"]) as r, rasterio.open(downloaded_s2["B03"]) as g, \
         rasterio.open(downloaded_s2["B02"]) as b, rasterio.open(downloaded_s2["B08"]) as nir:
        opt_arr = np.stack([r.read(1), g.read(1), b.read(1), nir.read(1)], axis=0).astype(np.float32)

    with rasterio.open(downloaded_s1["VV"]) as vv, rasterio.open(downloaded_s1["VH"]) as vh:
        sar_arr = np.stack([vv.read(1), vh.read(1)], axis=0).astype(np.float32)

    preprocessed = preprocess_multimodal_pair(opt_arr, sar_arr, inference_engine.device)
    inf_res = inference_engine.run_inference(preprocessed=preprocessed)

    print(f"  Inference Device: {inf_res['device']}")
    print(f"  Latency: {inf_res['latency_ms']} ms")
    print(f"  Output Mask Shape: {inf_res['seg_mask'].shape}")
    print(f"  Unique Semantic Classes Predicted: {np.unique(inf_res['seg_mask']).tolist()}")
    assert inf_res["seg_mask"].shape == (256, 256)
    print(">> PASSED: Dual-Stream Attention U-Net inference succeeded on provider rasters.")

    # -------------------------------------------------------------------------
    # TEST 11: Georeferenced Flood Mask GeoTIFF Export
    # -------------------------------------------------------------------------
    print("\n[TEST 11] Testing Georeferenced GeoTIFF Flood Mask Export...")
    out_flood_tif = os.path.join(str(SATELLITE_CACHE_DIR), f"flood_mask_{pair_res['analysis_id']}.tif")
    exported_path = inference_engine.export_geotiff_flood_mask(
        seg_mask=inf_res["seg_mask"],
        bounds=tuple(aoi_bbox),
        output_path=out_flood_tif,
    )
    assert os.path.exists(exported_path)
    flood_val = raster_validator.validate_raster_file(exported_path)
    print(f"  Exported GeoTIFF: {exported_path}")
    print(f"  Validation: Format={flood_val['format']}, Integrity={flood_val['integrity_status']}, CRS={flood_val.get('crs')}")
    assert flood_val["is_valid"] is True
    print(">> PASSED: Georeferenced GeoTIFF flood mask created and verified with rasterio.")

    # -------------------------------------------------------------------------
    # TEST 12: Corrupted & Truncated File Detection
    # -------------------------------------------------------------------------
    print("\n[TEST 12] Testing Corrupted/Empty File Detection in RasterValidator...")
    corrupt_file = os.path.join(str(SATELLITE_CACHE_DIR), "test_corrupt.tif")
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_TIFF_FILE_TRUNCATED")
    corrupt_val = raster_validator.validate_raster_file(corrupt_file)
    print(f"  Corrupted File Result: Integrity={corrupt_val['integrity_status']}, Valid={corrupt_val['is_valid']}")
    assert corrupt_val["is_valid"] is False
    os.remove(corrupt_file)
    print(">> PASSED: RasterValidator correctly detected corrupted/truncated raster.")

    print("\n" + "=" * 80)
    print("ALL 12 DUAL-PROVIDER ACCEPTANCE TESTS PASSED SUCCESSFULLY (100% PASS RATE)")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
