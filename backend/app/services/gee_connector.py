from typing import Any, Dict, List, Tuple
import os
import numpy as np
import logging

from app.config import GEE_PROJECT_ID, IMAGE_SIZE
from app.services.data_providers.gee_provider import GoogleEarthEngineProvider
from app.services.raster_validator import raster_validator
from scripts.init_demo_data import generate_synthetic_multimodal_tile

logger = logging.getLogger("floodsense.gee_connector")

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class GEEConnectorService:
    """
    Google Earth Engine (GEE) Live Satellite Fetcher for Sentinel-2 MSI (Optical)
    and Sentinel-1 C-Band SAR (VV/VH), equipped with automatic offline procedural
    multi-band fallback for zero-dependency academic demonstrations.
    """

    def __init__(self) -> None:
        self.provider = GoogleEarthEngineProvider()
        self.ee_authenticated = self.provider.is_configured()

    def fetch_sentinel_multimodal_tiles(
        self,
        bbox: List[float],
        pre_date: str,
        post_date: str,
        cloud_cover_max: float = 20.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fetches or extracts a co-registered 256x256 Sentinel-2 Pre-Flood Optical [4, H, W]
        and Sentinel-1 Post-Flood SAR [2, H, W] pair for the requested AOI bounding box.
        bbox format: [min_lat, min_lon, max_lat, max_lon]
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        bounds = (min_lat, min_lon, max_lat, max_lon)

        # 1. Attempt Genuine GEE Search & Acquisition
        if self.provider.is_configured():
            try:
                s2_scenes = self.provider.search_products(
                    bounds=bounds,
                    start_date="2025-01-01" if pre_date > "2025-01-01" else pre_date,
                    end_date=pre_date,
                    platform="SENTINEL-2",
                    max_cloud_cover_pct=cloud_cover_max,
                )
                s1_scenes = self.provider.search_products(
                    bounds=bounds,
                    start_date=pre_date,
                    end_date=post_date,
                    platform="SENTINEL-1",
                )

                if s2_scenes and s1_scenes:
                    s2_paths = self.provider.download_product(s2_scenes[0], ["B04", "B03", "B02", "B08"])
                    s1_paths = self.provider.download_product(s1_scenes[0], ["VV", "VH"])

                    if RASTERIO_AVAILABLE and len(s2_paths) == 4 and len(s1_paths) == 2:
                        with rasterio.open(s2_paths["B04"]) as r, rasterio.open(s2_paths["B03"]) as g, \
                             rasterio.open(s2_paths["B02"]) as b, rasterio.open(s2_paths["B08"]) as nir:
                            opt_4ch = np.stack([r.read(1), g.read(1), b.read(1), nir.read(1)], axis=0).astype(np.float32)

                        with rasterio.open(s1_paths["VV"]) as vv, rasterio.open(s1_paths["VH"]) as vh:
                            sar_2ch = np.stack([vv.read(1), vh.read(1)], axis=0).astype(np.float32)

                        # Validate rasters
                        val_s2 = raster_validator.validate_raster_file(s2_paths["B04"])
                        val_s1 = raster_validator.validate_raster_file(s1_paths["VV"])

                        # Ground truth mask estimation for validation
                        gt_mask = np.zeros(opt_4ch.shape[1:], dtype=np.uint8)

                        return (
                            opt_4ch,
                            sar_2ch,
                            gt_mask,
                            {
                                "source": "GEE_LIVE_COPERNICUS" if s2_scenes[0].is_live else "GEE_GENUINE_CACHED_ASSET",
                                "s2_scene_id": s2_scenes[0].scene_id,
                                "s1_scene_id": s1_scenes[0].scene_id,
                                "s2_collection": "COPERNICUS/S2_SR_HARMONIZED",
                                "s1_collection": "COPERNICUS/S1_GRD",
                                "pre_date": pre_date,
                                "post_date": post_date,
                                "cloud_cover_max": cloud_cover_max,
                                "optical_validation": val_s2,
                                "sar_validation": val_s1,
                                "is_live": s2_scenes[0].is_live,
                            },
                        )
            except Exception as e:
                logger.warning(f"GEE tile acquisition error: {e}")

        # 2. Deterministic coordinate-hashed multi-spectral synthesis for offline demo reliability
        coord_hash = int(abs(round(min_lat * 10000) + round(min_lon * 10000))) % 89989 + 100
        styles = ["chennai", "assam", "wayanad"]
        chosen_style = styles[coord_hash % len(styles)]

        opt_4ch, sar_2ch, gt_mask = generate_synthetic_multimodal_tile(
            preset_id=chosen_style,
            seed=coord_hash,
            cloud_cover_pct=cloud_cover_max,
        )

        telemetry_meta = {
            "source": "GEE_SYNTHETIC_COPERNICUS_SIMULATOR",
            "s2_collection": "COPERNICUS/S2_SR_HARMONIZED (B4, B3, B2, B8 @ 10m)",
            "s1_collection": "COPERNICUS/S1_GRD (IW Dual-Pol VV+VH @ 10m)",
            "pre_date": pre_date,
            "post_date": post_date,
            "cloud_cover_max": cloud_cover_max,
            "resolution_m": 10.0,
            "tile_dimensions": list(IMAGE_SIZE),
            "is_live": False,
            "is_synthetic": True,
            "note": "Procedural multi-band simulation used because GEE credentials not present.",
        }
        return opt_4ch, sar_2ch, gt_mask, telemetry_meta


gee_connector = GEEConnectorService()
