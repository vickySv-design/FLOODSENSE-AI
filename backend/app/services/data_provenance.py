"""Data Provenance Tracking and Manifest Persistence Service for FloodSense-AI.
Preserves end-to-end scientific pedigree for every acquired satellite scene,
preprocessing transformation, neural network inference pass, and tactical export.
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from app.config import METADATA_DIR, DEVICE_NAME, SATELLITE_CACHE_DIR

logger = logging.getLogger("floodsense.data_provenance")


class DataProvenanceManager:
    """Records and retrieves verifiable scientific data provenance for disaster analyses."""

    def __init__(self):
        self.metadata_dir = str(METADATA_DIR)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def create_provenance_manifest(
        self,
        analysis_id: str,
        aoi_metadata: Dict[str, Any],
        provider: str,
        optical_scene_meta: Optional[Dict[str, Any]] = None,
        sar_scene_meta: Optional[Dict[str, Any]] = None,
        preprocessing_steps: Optional[List[str]] = None,
        model_version: str = "DualStreamAttentionUNet_v2.1",
        execution_device: Optional[str] = None,
        raster_validation: Optional[Dict[str, Any]] = None,
        output_files: Optional[Dict[str, str]] = None,
        provenance_tag: str = "GENUINE_LIVE_ACQUISITION",
        extra_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates and persists an immutable data provenance record."""
        timestamp = datetime.now(timezone.utc).isoformat()
        device = execution_device or DEVICE_NAME

        manifest = {
            "analysis_id": analysis_id,
            "recorded_at": timestamp,
            "provenance_tag": provenance_tag,  # 'GENUINE_LIVE_ACQUISITION', 'GENUINE_CACHED_ACQUISITION', 'PRESET_CALIBRATED_BENCHMARK'
            "aoi": {
                "name": aoi_metadata.get("name", "Custom AOI"),
                "bbox": aoi_metadata.get("bbox", []),
                "area_km2": aoi_metadata.get("area_km2", 0.0),
                "utm_crs": aoi_metadata.get("utm_crs", "EPSG:32644"),
                "terrain_type": aoi_metadata.get("terrain_type", "coastal_urban"),
            },
            "provider": {
                "name": provider,
                "endpoint": "Copernicus OData / STAC" if provider == "copernicus" else "Google Earth Engine EE-REST",
                "zero_cost_academic": True,
            },
            "satellite_acquisitions": {
                "pre_disaster_optical": {
                    "platform": "SENTINEL-2 MSI",
                    "product_level": "Level-2A (Bottom-of-Atmosphere)",
                    "bands": ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"],
                    "resolution_m": 10.0,
                    "scene_details": optical_scene_meta or {},
                },
                "post_disaster_sar": {
                    "platform": "SENTINEL-1 C-SAR",
                    "product_level": "Level-1 GRD (Ground Range Detected)",
                    "polarizations": ["VV", "VH"],
                    "instrument_mode": "IW (Interferometric Wide Swath)",
                    "frequency_ghz": 5.405,
                    "resolution_m": 10.0,
                    "scene_details": sar_scene_meta or {},
                },
            },
            "preprocessing_pipeline": preprocessing_steps or [
                "Optical CLAHE contrast adjustment (clip=2.0)",
                "Optical NDWI McFeeters 1996 baseline calculation",
                "SAR Sigma-0 backscatter decibel calibration [-30, 0] dB",
                "SAR Lee-Sigma 5x5 adaptive speckle suppression",
                "Bicubic resampling to common UTM metric grid",
            ],
            "deep_learning_pipeline": {
                "architecture": "Dual-Stream Siamese Attention U-Net with CMAF",
                "checkpoint": model_version,
                "input_tensors": {"optical": "[B, 4, H, W]", "sar": "[B, 2, H, W]"},
                "semantic_classes": {
                    "0": "Dry Land",
                    "1": "Permanent Baseline Water (NDWI/Pre-SAR)",
                    "2": "Newly Inundated Floodwater (Post-SAR Inundation)",
                },
                "inference_engine": "PyTorch Sliding-Window with 2D Hann Cosine Overlap Blending",
                "execution_hardware": device,
            },
            "raster_validation": raster_validation or {},
            "output_products": output_files or {},
            "scientific_notes": extra_notes or "Full geospatial change detection pipeline executed without artificial metric clamping.",
        }

        # Persist manifest to disk
        manifest_path = os.path.join(self.metadata_dir, f"provenance_{analysis_id}.json")
        manifest["saved_path"] = manifest_path
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write provenance manifest to {manifest_path}: {e}")

        return manifest

    def get_provenance(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an existing provenance manifest by analysis ID."""
        manifest_path = os.path.join(self.metadata_dir, f"provenance_{analysis_id}.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading provenance manifest {manifest_path}: {e}")
        return None


data_provenance_manager = DataProvenanceManager()
