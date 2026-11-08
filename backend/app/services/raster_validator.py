"""Raster Validator Service for FloodSense-AI.
Inspects and validates genuine satellite raster files (GeoTIFF, NPY, NPZ)
using rasterio and numpy, calculating dimensions, CRS, affine transforms,
band statistics, and integrity verifications.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger("floodsense.raster_validator")

try:
    import rasterio
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class RasterValidator:
    """Verifies physical raster integrity, band distributions, and georeferencing metadata."""

    @staticmethod
    def validate_raster_file(file_path: str) -> Dict[str, Any]:
        """
        Thoroughly inspects a raster file on disk.
        Returns detailed spatial metadata and band statistics.
        """
        if not os.path.exists(file_path):
            return {
                "file_path": file_path,
                "exists": False,
                "is_valid": False,
                "integrity_status": "MISSING",
                "error": f"File does not exist: {file_path}",
            }

        file_size = os.path.getsize(file_path)
        if file_size < 128:
            return {
                "file_path": file_path,
                "exists": True,
                "file_size_bytes": file_size,
                "is_valid": False,
                "integrity_status": "EMPTY_OR_TRUNCATED",
                "error": f"File size too small ({file_size} bytes) to contain valid raster pixels.",
            }

        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".tif", ".tiff"] and RASTERIO_AVAILABLE:
            return RasterValidator._validate_geotiff(file_path, file_size)
        elif ext in [".npy", ".npz"]:
            return RasterValidator._validate_numpy(file_path, file_size)
        else:
            # Fallback inspection
            return RasterValidator._validate_fallback(file_path, file_size)

    @staticmethod
    def _validate_geotiff(file_path: str, file_size: int) -> Dict[str, Any]:
        """Validates GeoTIFF with rasterio."""
        try:
            with rasterio.open(file_path) as src:
                width = src.width
                height = src.height
                band_count = src.count
                crs = str(src.crs) if src.crs else "UNSPECIFIED"
                transform = [float(x) for x in list(src.transform)[:6]]
                bounds = {
                    "left": float(src.bounds.left),
                    "bottom": float(src.bounds.bottom),
                    "right": float(src.bounds.right),
                    "top": float(src.bounds.top),
                }
                nodata = float(src.nodata) if src.nodata is not None else None
                dtypes = [str(d) for d in src.dtypes]

                band_stats: Dict[int, Dict[str, Any]] = {}
                all_zeros = True
                has_nans = False

                for b in range(1, band_count + 1):
                    band_arr = src.read(b)
                    valid_mask = np.isfinite(band_arr)
                    if nodata is not None:
                        valid_mask = valid_mask & (band_arr != nodata)

                    valid_count = int(np.count_nonzero(valid_mask))
                    total_pixels = band_arr.size
                    valid_pct = round((valid_count / max(total_pixels, 1)) * 100.0, 2)

                    if valid_count > 0:
                        valid_vals = band_arr[valid_mask]
                        b_min = float(np.min(valid_vals))
                        b_max = float(np.max(valid_vals))
                        b_mean = float(np.mean(valid_vals))
                        b_std = float(np.std(valid_vals))
                        non_zero = int(np.count_nonzero(valid_vals != 0))
                        non_zero_pct = round((non_zero / max(valid_count, 1)) * 100.0, 2)

                        if non_zero > 0:
                            all_zeros = False
                    else:
                        b_min, b_max, b_mean, b_std, non_zero_pct = 0.0, 0.0, 0.0, 0.0, 0.0

                    if np.any(np.isnan(band_arr)):
                        has_nans = True

                    band_stats[b] = {
                        "band_index": b,
                        "dtype": dtypes[b - 1] if b - 1 < len(dtypes) else "unknown",
                        "valid_pixel_pct": valid_pct,
                        "non_zero_pct": non_zero_pct,
                        "min": round(b_min, 4),
                        "max": round(b_max, 4),
                        "mean": round(b_mean, 4),
                        "std": round(b_std, 4),
                    }

                integrity = "VERIFIED"
                if all_zeros:
                    integrity = "ALL_ZEROS_SUSPECT"
                elif has_nans and any(s["valid_pixel_pct"] < 50 for s in band_stats.values()):
                    integrity = "HIGH_NODATA_WARNING"

                return {
                    "file_path": file_path,
                    "format": "GeoTIFF",
                    "exists": True,
                    "is_valid": not all_zeros,
                    "integrity_status": integrity,
                    "file_size_bytes": file_size,
                    "file_size_mb": round(file_size / (1024 * 1024), 3),
                    "dimensions": {"width": width, "height": height},
                    "band_count": band_count,
                    "crs": crs,
                    "affine_transform": transform,
                    "bounds": bounds,
                    "nodata_value": nodata,
                    "band_statistics": band_stats,
                }
        except Exception as e:
            logger.error(f"GeoTIFF validation error for {file_path}: {e}")
            return {
                "file_path": file_path,
                "format": "GeoTIFF",
                "exists": True,
                "is_valid": False,
                "integrity_status": "CORRUPTED",
                "file_size_bytes": file_size,
                "error": str(e),
            }

    @staticmethod
    def _validate_numpy(file_path: str, file_size: int) -> Dict[str, Any]:
        """Validates NPY or NPZ raster array."""
        try:
            if file_path.endswith(".npz"):
                with np.load(file_path) as data:
                    first_key = list(data.keys())[0]
                    arr = data[first_key]
            else:
                arr = np.load(file_path)

            shape = list(arr.shape)
            dtype = str(arr.dtype)
            total = arr.size
            non_zero = int(np.count_nonzero(arr != 0))
            non_zero_pct = round((non_zero / max(total, 1)) * 100.0, 2)

            return {
                "file_path": file_path,
                "format": "NumPy_Array",
                "exists": True,
                "is_valid": non_zero > 0,
                "integrity_status": "VERIFIED" if non_zero > 0 else "ALL_ZEROS",
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 3),
                "shape": shape,
                "dtype": dtype,
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "non_zero_pct": non_zero_pct,
            }
        except Exception as e:
            return {
                "file_path": file_path,
                "format": "NumPy_Array",
                "exists": True,
                "is_valid": False,
                "integrity_status": "CORRUPTED",
                "file_size_bytes": file_size,
                "error": str(e),
            }

    @staticmethod
    def _validate_fallback(file_path: str, file_size: int) -> Dict[str, Any]:
        """Fallback validation checking basic readability."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(16)
            is_tiff = header.startswith(b"II*\x00") or header.startswith(b"MM\x00*")
            return {
                "file_path": file_path,
                "format": "TIFF" if is_tiff else "BINARY",
                "exists": True,
                "is_valid": is_tiff,
                "integrity_status": "VERIFIED_HEADER" if is_tiff else "UNKNOWN_BINARY",
                "file_size_bytes": file_size,
            }
        except Exception as e:
            return {
                "file_path": file_path,
                "exists": True,
                "is_valid": False,
                "integrity_status": "UNREADABLE",
                "error": str(e),
            }


raster_validator = RasterValidator()

