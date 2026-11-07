import argparse
import json
from pathlib import Path
import sys

# Ensure backend root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import DISASTER_PRESETS_METADATA, OSM_CACHE_DIR
from app.services.osm_service import osm_service


def fetch_and_cache_all_presets(force: bool = False):
    print("=" * 78)
    print("  FLOODSENSE-AI // OPENSTREETMAP ROAD & BUILDING FETCH & CACHE SCRIPT")
    print(f"  Target Cache Directory: {OSM_CACHE_DIR}")
    print("=" * 78)

    summary = []

    for preset_id, meta in DISASTER_PRESETS_METADATA.items():
        name = meta["name"]
        bbox = meta["bbox"]
        print(f"\n[*] Processing Preset: {preset_id} ({name})")
        print(f"    BBox: [{bbox[0]}N, {bbox[1]}E] to [{bbox[2]}N, {bbox[3]}E]")

        t0 = False
        roads_geo, blds_geo, from_cache, warning = osm_service.fetch_and_cache_osm_data(
            preset_id, bbox, force_refresh=force
        )

        roads_path, blds_path = osm_service.get_preset_cache_paths(preset_id)
        roads_count = len(roads_geo.get("features", []))
        blds_count = len(blds_geo.get("features", []))

        status_str = "CACHED" if from_cache else "DOWNLOADED"
        if warning:
            status_str = "WARNING"
            print(f"    [!] Warning: {warning}")
        else:
            print(f"    [OK] Road Features:     {roads_count:,} ({roads_path.name})")
            print(f"    [OK] Building Features: {blds_count:,} ({blds_path.name})")
            print(f"    [OK] Status:            {status_str}")

        summary.append({
            "preset_id": preset_id,
            "preset_name": name,
            "road_features": roads_count,
            "building_features": blds_count,
            "status": status_str,
            "roads_cache_file": str(roads_path),
            "buildings_cache_file": str(blds_path),
        })

    print("\n" + "=" * 78)
    print("  SUMMARY OF OPENSTREETMAP DATA ACQUISITION:")
    print("=" * 78)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and cache real OSM roads and building footprints.")
    parser.add_argument("--all-presets", action="store_true", help="Fetch OSM data for all 3 disaster presets.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cache exists.")
    args = parser.parse_args()

    fetch_and_cache_all_presets(force=args.force)

