import os
from pathlib import Path
import torch

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRESETS_DIR = DATA_DIR / "presets"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
OSM_CACHE_DIR = DATA_DIR / "osm_cache"
AOI_CACHE_DIR = DATA_DIR / "aoi_cache"
SATELLITE_CACHE_DIR = DATA_DIR / "satellite_cache"
DEM_CACHE_DIR = DATA_DIR / "dem_cache"
RAINFALL_CACHE_DIR = DATA_DIR / "rainfall_cache"
PROCESSED_CACHE_DIR = DATA_DIR / "processed_cache"
METADATA_DIR = DATA_DIR / "metadata"
DOWNLOADS_DIR = DATA_DIR / "downloads"
SCENARIOS_DIR = DATA_DIR / "scenarios"

for d in [
    PRESETS_DIR,
    CHECKPOINTS_DIR,
    OSM_CACHE_DIR,
    AOI_CACHE_DIR,
    SATELLITE_CACHE_DIR,
    DEM_CACHE_DIR,
    RAINFALL_CACHE_DIR,
    PROCESSED_CACHE_DIR,
    METADATA_DIR,
    DOWNLOADS_DIR,
    SCENARIOS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

DATA_MODE = os.environ.get("DATA_MODE", "hybrid").lower()
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"

MODEL_WEIGHTS_PATH = CHECKPOINTS_DIR / "dual_stream_attention_unet.pth"

# Hardware Acceleration Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = (
    f"CUDA ({torch.cuda.get_device_name(0)})"
    if torch.cuda.is_available()
    else "CPU Engine (PyTorch Optimized)"
)

# Remote Sensing & Spatial Telemetry Constants
IMAGE_SIZE = (256, 256)
SPATIAL_RESOLUTION_M = 10.0  # Sentinel-1/2 10m x 10m pixel resolution
PIXEL_AREA_M2 = SPATIAL_RESOLUTION_M * SPATIAL_RESOLUTION_M  # 100 m^2
PIXEL_AREA_KM2 = PIXEL_AREA_M2 / 1_000_000.0  # 0.0001 km^2 per pixel
HIGH_RISK_BUFFER_PIXELS = 10  # 10 pixels * 10m = 100m High-Risk Buffer Zone

# Semantic Segmentation Classes
CLASS_DRY_LAND = 0
CLASS_PERMANENT_WATER = 1
CLASS_NEW_FLOOD = 2
NUM_CLASSES = 3

# Rescue Routing Constants
GRID_NODES_PER_SIDE = 12  # 12x12 planar road network graph
AVERAGE_RESCUE_SPEED_KMH = 28.0  # Tactical amphibious/high-clearance rescue vehicle speed

# Google Earth Engine Configuration
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "floodsense-ai-capstone")
GEE_ENABLED = os.getenv("GEE_ENABLED", "auto")

# Pre-Loaded Disaster Case Study Metadata
DISASTER_PRESETS_METADATA = {
    "chennai_monsoon_2026": {
        "id": "chennai_monsoon_2026",
        "name": "Chennai Adyar River Catchment & Coastal Bay of Bengal Estuary",
        "region": "Tamil Nadu, India",
        "event_date": "2025-12-04",
        "pre_date": "2025-11-10",
        "post_date": "2025-12-05",
        "bbox": [12.980, 80.190, 13.060, 80.290],  # [min_lat, min_lon, max_lat, max_lon] covering full Adyar catchment & coast
        "center": [13.020, 80.240],
        "zoom": 13,
        "sensor_pair": "Sentinel-2B MSI + Sentinel-1A C-Band SAR (VV/VH)",
        "description": "Comprehensive coastal riverine basin covering upstream Adyar catchment, urban river corridor, estuary, and adjacent Bay of Bengal coastline.",
        "population_density_factor": 42,
        "terrain_type": "coastal_urban",
        "has_ocean_coastline": True,
        "historical_metrics": {
            "miou": 0.914,
            "f1_score": 0.948,
            "precision": 0.956,
            "recall": 0.941,
        },
    },
    "assam_brahmaputra_surge": {
        "id": "assam_brahmaputra_surge",
        "name": "Assam Brahmaputra Valley Monsoon Inundation",
        "region": "Guwahati & Kamrup Basin, Assam, India",
        "event_date": "2025-07-18",
        "pre_date": "2025-05-12",
        "post_date": "2025-07-19",
        "bbox": [26.140, 91.700, 26.190, 91.760],
        "center": [26.165, 91.730],
        "zoom": 14,
        "sensor_pair": "Sentinel-2A MSI + Sentinel-1A IW GRD SAR",
        "description": "Massive braided river channel expansion and embankment breach isolating rural settlements and agricultural floodplains.",
        "population_density_factor": 28,
        "terrain_type": "river_floodplain",
        "has_ocean_coastline": False,
        "historical_metrics": {
            "miou": 0.928,
            "f1_score": 0.959,
            "precision": 0.962,
            "recall": 0.955,
        },
    },
    "wayanad_kerala_flashflood": {
        "id": "wayanad_kerala_flashflood",
        "name": "Kerala Kabini Basin Flash Flood & Landslide Surge",
        "region": "Wayanad District, Kerala, India",
        "event_date": "2025-08-02",
        "pre_date": "2025-06-20",
        "post_date": "2025-08-03",
        "bbox": [11.680, 76.100, 11.730, 76.150],
        "center": [11.705, 76.125],
        "zoom": 14,
        "sensor_pair": "Sentinel-2B Optical + Sentinel-1A Dual-Pol SAR",
        "description": "Steep Western Ghats terrain dendritic channel flash flooding with acute highway severance under heavy monsoon cloud cover.",
        "population_density_factor": 22,
        "terrain_type": "mountainous_catchment",
        "has_ocean_coastline": False,
        "historical_metrics": {
            "miou": 0.897,
            "f1_score": 0.936,
            "precision": 0.944,
            "recall": 0.929,
        },
    },
}

