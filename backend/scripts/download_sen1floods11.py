"""Automated Zero-Cost Downloader for Official Sen1Floods11 v1.1 Hand-Labeled Dataset.

Downloads:
- 446 Hand-Labeled Water Masks (LabelHand/)
- 446 Sentinel-1 IW GRD SAR GeoTIFFs (S1Hand/)
- 446 Sentinel-2 Level-1C MSI GeoTIFFs (S2Hand/)
- 446 JRC Permanent Water References (JRCWaterHand/)
- 4 Official Split Manifest CSVs
- 1 Event Metadata GeoJSON

Total download volume: ~1.63 GB (1,789 files).
Supports automatic resumption, parallel downloads, and verification.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://storage.googleapis.com/sen1floods11"
GITHUB_META_URL = "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/Sen1Floods11_Metadata.geojson"

DEST_ROOT = BACKEND_ROOT / "data" / "official_sen1floods11"
RAW_DIR = DEST_ROOT / "raw"
SPLITS_DIR = DEST_ROOT / "splits"
META_DIR = DEST_ROOT / "metadata"

SPLIT_FILES = [
    "flood_train_data.csv",
    "flood_valid_data.csv",
    "flood_test_data.csv",
    "flood_bolivia_data.csv",
]

LAYER_FOLDERS = {
    "LabelHand": ("v1.1/data/flood_events/HandLabeled/LabelHand", "_LabelHand.tif"),
    "S1Hand": ("v1.1/data/flood_events/HandLabeled/S1Hand", "_S1Hand.tif"),
    "S2Hand": ("v1.1/data/flood_events/HandLabeled/S2Hand", "_S2Hand.tif"),
    "JRCWaterHand": ("v1.1/data/flood_events/HandLabeled/JRCWaterHand", "_JRCWaterHand.tif"),
}


def download_file(url: str, dest_path: Path, max_retries: int = 3) -> tuple[str, bool, int]:
    """Downloads a file if not already present with valid size. Returns (filename, success, bytes)."""
    if dest_path.exists() and dest_path.stat().st_size > 500:
        return (dest_path.name, True, dest_path.stat().st_size)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FloodSense-AI/1.0 (Academic Research; Remote Sensing)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            if temp_path.exists() and temp_path.stat().st_size > 0:
                temp_path.replace(dest_path)
                return (dest_path.name, True, dest_path.stat().st_size)
        except Exception as e:
            if attempt == max_retries - 1:
                if temp_path.exists():
                    temp_path.unlink()
                print(f"  [ERROR] Failed to download {url}: {e}", file=sys.stderr)
                return (dest_path.name, False, 0)
            time.sleep(1.0 + attempt * 2.0)

    return (dest_path.name, False, 0)


def run_download(workers: int = 8) -> None:
    print("=" * 78)
    print("  FLOODSENSE-AI: OFFICIAL SEN1FLOODS11 v1.1 DATASET DOWNLOADER")
    print("  Scope: 446 Hand-Labeled Multi-Modal Chips (S1 + S2 + Label + JRC)")
    print(f"  Destination: {DEST_ROOT}")
    print("=" * 78)

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download Split Files
    print("\n[PHASE 1/3] Downloading Official Split Manifests & Event Metadata...")
    all_chips = set()

    for sf in SPLIT_FILES:
        url = f"{BASE_URL}/v1.1/splits/flood_handlabeled/{sf}"
        dest = SPLITS_DIR / sf
        _, ok, sz = download_file(url, dest)
        if ok:
            print(f"  ✔ Downloaded split manifest: {sf} ({sz:,} bytes)")
            with open(dest, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        s1_name = line.split(",")[0].strip()
                        stem = s1_name.replace("_S1Hand.tif", "")
                        all_chips.add(stem)

    # Download Metadata GeoJSON
    meta_dest = META_DIR / "Sen1Floods11_Metadata.geojson"
    _, ok, sz = download_file(GITHUB_META_URL, meta_dest)
    if ok:
        print(f"  ✔ Downloaded event metadata: Sen1Floods11_Metadata.geojson ({sz:,} bytes)")

    chip_list = sorted(all_chips)
    print(f"\n  Total Unique Hand-Labeled Chips Identified: {len(chip_list)} chips.")
    assert len(chip_list) == 446, f"Expected 446 chips, found {len(chip_list)}"

    # 2. Build Queue of All GeoTIFF Files
    print(f"\n[PHASE 2/3] Preparing Download Queue ({len(chip_list) * 4} GeoTIFFs across 4 layers)...")
    download_tasks = []

    for layer_name, (prefix, suffix) in LAYER_FOLDERS.items():
        layer_dest = RAW_DIR / layer_name
        layer_dest.mkdir(parents=True, exist_ok=True)
        for chip in chip_list:
            filename = f"{chip}{suffix}"
            url = f"{BASE_URL}/{prefix}/{filename}"
            dest_file = layer_dest / filename
            download_tasks.append((url, dest_file))

    print(f"  Queued {len(download_tasks)} files. Executing parallel download with {workers} worker threads...")

    # 3. Parallel Execution
    t0 = time.perf_counter()
    completed = 0
    total_bytes = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_file, url, dest): dest.name for url, dest in download_tasks}
        for future in as_completed(futures):
            fname, success, nbytes = future.result()
            completed += 1
            if success:
                total_bytes += nbytes
            else:
                failed += 1

            if completed % 100 == 0 or completed == len(download_tasks):
                mb = total_bytes / (1024 * 1024)
                rate = mb / max(time.perf_counter() - t0, 0.01)
                print(f"  Progress: [{completed:>4}/{len(download_tasks)}] ({completed/len(download_tasks)*100:>5.1f}%) | Downloaded: {mb:>7.1f} MB | Speed: {rate:>5.1f} MB/s")

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 78)
    print(f"  ACQUISITION COMPLETE in {elapsed:.1f}s")
    print(f"  Files Successfully Verified: {completed - failed} / {len(download_tasks)}")
    print(f"  Total Data Downloaded: {total_bytes / (1024*1024):.2f} MB (~{total_bytes / (1024**3):.2f} GB)")
    print(f"  Failed Downloads: {failed}")
    print(f"  Destination: {DEST_ROOT}")
    print("=" * 78)


if __name__ == "__main__":
    run_download()

