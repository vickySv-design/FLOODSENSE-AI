import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.georef import GeoSpatialTransform

def test_georef():
    print("[TEST] Running Georeferencing & Projection Validation...")
    # Chennai bounding box [min_lat, min_lon, max_lat, max_lon]
    bbox = [13.00, 80.20, 13.08, 80.28]
    gt = GeoSpatialTransform(bbox, width=256, height=256)

    # 1. Coordinate round-trip test
    col_orig, row_orig = 128.5, 64.2
    lon, lat = gt.pixel_to_lonlat(col_orig, row_orig)
    col_back, row_back = gt.lonlat_to_pixel(lon, lat)
    assert abs(col_orig - col_back) < 1e-4, f"Col roundtrip failed: {col_orig} vs {col_back}"
    assert abs(row_orig - row_back) < 1e-4, f"Row roundtrip failed: {row_orig} vs {row_back}"
    print(f"  [PASS] Pixel <-> Lon/Lat Roundtrip: ({col_orig}, {row_orig}) -> ({lon}, {lat}) -> ({col_back:.2f}, {row_back:.2f})")

    # 2. UTM metric projection test
    e, n = gt.lonlat_to_utm(80.24, 13.04)
    lon_rt, lat_rt = gt.utm_to_lonlat(e, n)
    assert abs(lon_rt - 80.24) < 1e-4, f"UTM roundtrip lon failed: {lon_rt}"
    assert abs(lat_rt - 13.04) < 1e-4, f"UTM roundtrip lat failed: {lat_rt}"
    print(f"  [PASS] WGS84 <-> UTM EPSG:{gt.utm_epsg} Roundtrip: (80.24, 13.04) -> ({e:.1f}m, {n:.1f}m)")

    # 3. Polygon metric area test
    # 0.01 deg x 0.01 deg square at ~13 deg N is ~1100m x ~1080m ~= 1.18 km^2
    square_poly = [
        [80.24, 13.04],
        [80.25, 13.04],
        [80.25, 13.05],
        [80.24, 13.05],
        [80.24, 13.04]
    ]
    area_m2 = gt.compute_polygon_area_m2(square_poly)
    area_km2 = area_m2 / 1e6
    assert 1.0 < area_km2 < 1.4, f"Unexpected area: {area_km2} km2"
    print(f"  [PASS] Metric Polygon Area: {area_m2:.1f} m2 ({area_km2:.3f} km2)")

    # 4. Bilinear sampling test
    raster = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    sample_val = gt.sample_raster_at_lonlat(raster, lon, lat, method="bilinear")
    expected_val = row_orig * 256 + col_orig
    assert abs(sample_val - expected_val) < 2.0, f"Sample mismatch: {sample_val} vs {expected_val}"
    print(f"  [PASS] Raster Sub-Pixel Sampling: sample={sample_val:.2f}, expected={expected_val:.2f}")

    print("\nALL GEOREFERENCING TESTS PASSED (4/4)!")

if __name__ == "__main__":
    test_georef()
