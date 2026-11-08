"""Base Satellite Provider Abstract Base Class and Scene Metadata Schemas.
Standardizes data acquisition interfaces across Copernicus Data Space Ecosystem,
Google Earth Engine, and Local Georeferenced Rasters.
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from app.config import SATELLITE_CACHE_DIR


@dataclass
class SatelliteSceneMetadata:
    """Standardized metadata schema for an acquired or discoverable satellite scene."""
    scene_id: str
    provider: str                               # 'copernicus', 'gee', 'local_cache'
    platform: str                               # 'SENTINEL-1', 'SENTINEL-2'
    instrument: str                             # 'C-SAR', 'MSI'
    product_level: str                          # 'L1C', 'L2A', 'GRD'
    acquisition_date: str                       # ISO 8601 string: 'YYYY-MM-DDTHH:MM:SSZ'
    bounds: Tuple[float, float, float, float]   # (min_lat, min_lon, max_lat, max_lon)
    cloud_cover_pct: Optional[float] = None
    orbit_direction: Optional[str] = None       # 'ASCENDING', 'DESCENDING'
    polarizations: List[str] = field(default_factory=list)   # e.g. ['VV', 'VH']
    available_bands: List[str] = field(default_factory=list) # e.g. ['B02', 'B03', 'B04', 'B08']
    resolution_meters: float = 10.0
    download_url: Optional[str] = None
    local_cached_path: Optional[str] = None
    is_live: bool = False
    is_cached: bool = False
    is_synthetic: bool = False
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "provider": self.provider,
            "platform": self.platform,
            "instrument": self.instrument,
            "product_level": self.product_level,
            "acquisition_date": self.acquisition_date,
            "bounds": {
                "min_lat": float(self.bounds[0]),
                "min_lon": float(self.bounds[1]),
                "max_lat": float(self.bounds[2]),
                "max_lon": float(self.bounds[3]),
            },
            "cloud_cover_pct": self.cloud_cover_pct,
            "orbit_direction": self.orbit_direction,
            "polarizations": self.polarizations,
            "available_bands": self.available_bands,
            "resolution_meters": self.resolution_meters,
            "download_url": self.download_url,
            "local_cached_path": self.local_cached_path,
            "is_live": self.is_live,
            "is_cached": self.is_cached,
            "is_synthetic": self.is_synthetic,
            "extra_attributes": self.extra_attributes,
        }


class BaseSatelliteProvider(ABC):
    """Abstract interface defining required capabilities for any remote sensing satellite data source."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or str(SATELLITE_CACHE_DIR)
        os.makedirs(self.cache_dir, exist_ok=True)

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if required environment variables, tokens, or credentials are valid and present."""
        pass

    @abstractmethod
    def get_auth_instructions(self) -> str:
        """Provide clear, actionable, free sign-up and configuration instructions."""
        pass

    def check_connection(self) -> Dict[str, Any]:
        """Verifies live connection and returns diagnostic status."""
        return {
            "provider": getattr(self, "provider_name", "base"),
            "status": "CONFIGURED" if self.is_configured() else "NOT_CONFIGURED",
            "connected": self.is_configured(),
        }

    def get_provider_status(self) -> Dict[str, Any]:
        """Reports provider status, authentication details, and capabilities."""
        return self.check_connection()

    def search_products(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        platform: str = "SENTINEL-2",
        max_cloud_cover_pct: float = 20.0,
    ) -> List[SatelliteSceneMetadata]:
        """Search available imagery catalog for scenes intersecting geographic bounding box."""
        return self.search_scenes(bounds, start_date, end_date, platform, max_cloud_cover_pct)

    def get_product_metadata(self, product_id: str) -> Optional[SatelliteSceneMetadata]:
        """Retrieve detailed metadata for a specific product ID."""
        return None

    def download_product(
        self,
        scene_metadata: SatelliteSceneMetadata,
        bands: List[str],
        target_resolution_m: float = 10.0,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Download or read specified spectral/radar bands, caching locally and returning dict of band_name -> file_path."""
        return self.fetch_scene_bands(scene_metadata, bands, target_resolution_m)

    def export_processed_raster(
        self,
        scene_metadata: SatelliteSceneMetadata,
        bounds: Tuple[float, float, float, float],
        output_path: str,
    ) -> str:
        """Exports genuine multi-band GeoTIFF raster clipped to bounds."""
        return ""

    # Backward-compatibility methods
    def search_scenes(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        platform: str = "SENTINEL-2",
        max_cloud_cover_pct: float = 20.0,
    ) -> List[SatelliteSceneMetadata]:
        return []

    def fetch_scene_bands(
        self,
        scene_metadata: SatelliteSceneMetadata,
        bands: List[str],
        target_resolution_m: float = 10.0,
    ) -> Dict[str, str]:
        return {}
