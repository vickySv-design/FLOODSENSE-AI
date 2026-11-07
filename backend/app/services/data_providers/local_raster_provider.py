"""Local GeoTIFF / Raster Ingestion Provider.
Parses local GeoTIFF and multi-band TIFF files directly using tifffile, PIL, and OpenCV.
Extracts affine bounds, CRS metadata, band counts, and loads multi-channel tensors.
"""
import os
import glob
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import tifffile  # type: ignore
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

from PIL import Image

from .base_provider import BaseSatelliteProvider, SatelliteSceneMetadata

logger = logging.getLogger("floodsense.local_raster")


class LocalRasterProvider(BaseSatelliteProvider):
    """Local raster dataset provider for offline verification, user uploads, and local GeoTIFF files."""

    def __init__(self, data_root: Optional[str] = None):
        super().__init__()
        self.data_root = data_root or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
        )

    def is_configured(self) -> bool:
        return os.path.exists(self.data_root)

    def get_auth_instructions(self) -> str:
        return (
            "Local Raster Provider does not require online credentials.\n"
            f"Place GeoTIFF or multi-band TIFF files into: {self.data_root}/local_rasters/\n"
            "Supported layouts: 4-band Optical (RGB+NIR), 2-band SAR (VV+VH), or separate single-band GeoTIFFs."
        )

    def scan_local_files(self) -> List[Dict[str, Any]]:
        """Scans data root for valid raster files."""
        patterns = ["**/*.tif", "**/*.tiff", "**/*.npy"]
        found = []
        for pat in patterns:
            for p in glob.glob(os.path.join(self.data_root, pat), recursive=True):
                stat = os.stat(p)
                found.append({
                    "path": p,
                    "filename": os.path.basename(p),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "extension": os.path.splitext(p)[1].lower(),
                })
        return found

    def inspect_raster_file(self, file_path: str) -> Dict[str, Any]:
        """Extracts shape, dtype, band count, and GeoTIFF tags if available."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Raster file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".npy":
            arr = np.load(file_path)
            return {
                "format": "NUMPY_ARRAY",
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "bands": arr.shape[0] if arr.ndim == 3 else 1,
                "is_geotiff": False,
            }

        if TIFFFILE_AVAILABLE and ext in [".tif", ".tiff"]:
            try:
                with tifffile.TiffFile(file_path) as tf:
                    page = tf.pages[0]
                    geotags = tf.geotiff_metadata if hasattr(tf, "geotiff_metadata") else None
                    return {
                        "format": "GEOTIFF" if geotags else "TIFF",
                        "shape": list(page.shape),
                        "dtype": str(page.dtype),
                        "bands": len(tf.pages) if len(page.shape) == 2 and len(tf.pages) > 1 else (page.shape[-1] if len(page.shape) == 3 else 1),
                        "is_geotiff": bool(geotags),
                        "geotiff_tags": str(geotags)[:200] if geotags else None,
                    }
            except Exception as e:
                logger.warning(f"tifffile failed on {file_path}: {e}")

        # PIL fallback
        with Image.open(file_path) as img:
            return {
                "format": "IMAGE",
                "shape": [img.height, img.width],
                "mode": img.mode,
                "bands": len(img.getbands()),
                "is_geotiff": False,
            }

    def read_raster(self, file_path: str) -> np.ndarray:
        """Reads raster into a float32 NumPy array with channel-first layout [C, H, W]."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".npy":
            arr = np.load(file_path).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3 and arr.shape[-1] in [2, 3, 4] and arr.shape[0] not in [2, 3, 4]:
                arr = np.transpose(arr, (2, 0, 1))
            return arr

        if TIFFFILE_AVAILABLE and ext in [".tif", ".tiff"]:
            try:
                data = tifffile.imread(file_path).astype(np.float32)
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                elif data.ndim == 3 and data.shape[-1] in [2, 3, 4] and data.shape[0] not in [2, 3, 4]:
                    data = np.transpose(data, (2, 0, 1))
                return data
            except Exception as e:
                logger.warning(f"tifffile read failed on {file_path}: {e}")

        with Image.open(file_path) as img:
            arr = np.array(img).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                arr = np.transpose(arr, (2, 0, 1))
            return arr

    def search_scenes(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        platform: str = "LOCAL",
        max_cloud_cover_pct: float = 20.0,
    ) -> List[SatelliteSceneMetadata]:
        files = self.scan_local_files()
        scenes = []
        for f in files:
            scenes.append(SatelliteSceneMetadata(
                scene_id=os.path.splitext(f["filename"])[0],
                provider="local_geotiff",
                platform=platform,
                instrument="LOCAL_RASTER",
                product_level="PROCESSED",
                acquisition_date=f"{start_date}T00:00:00Z",
                bounds=bounds,
                resolution_meters=10.0,
                local_cached_path=f["path"],
                extra_attributes=f,
            ))
        return scenes

    def fetch_scene_bands(
        self,
        scene_metadata: SatelliteSceneMetadata,
        bands: List[str],
        target_resolution_m: float = 10.0,
    ) -> Dict[str, str]:
        path = scene_metadata.local_cached_path or ""
        return {b: path for b in bands}
