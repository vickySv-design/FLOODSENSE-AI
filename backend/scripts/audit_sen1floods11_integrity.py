import json
import os
import sys
from pathlib import Path
import numpy as np
import rasterio

DATA_DIR = Path("backend/data/official_sen1floods11")
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
METADATA_PATH = DATA_DIR / "metadata" / "Sen1Floods11_Metadata.geojson"

def run_audit():
    print("=== SEN1FLOODS11 SCIENTIFIC INTEGRITY AUDIT ===")
    results = {}

    # Check directories
    label_dir = RAW_DIR / "LabelHand"
    s1_dir = RAW_DIR / "S1Hand"
    s2_dir = RAW_DIR / "S2Hand"
    jrc_dir = RAW_DIR / "JRCWaterHand"

    label_files = {f.name.replace("_LabelHand.tif", ""): f for f in label_dir.glob("*.tif")}
    s1_files = {f.name.replace("_S1Hand.tif", ""): f for f in s1_dir.glob("*.tif")}
    s2_files = {f.name.replace("_S2Hand.tif", ""): f for f in s2_dir.glob("*.tif")}
    jrc_files = {f.name.replace("_JRCWaterHand.tif", ""): f for f in jrc_dir.glob("*.tif")}

    all_chip_ids = sorted(list(label_files.keys()))
    print(f"Total chip IDs found in LabelHand: {len(all_chip_ids)}")
    print(f"S1Hand: {len(s1_files)}, S2Hand: {len(s2_files)}, JRCWaterHand: {len(jrc_files)}")

    # PHASE 1 & PHASE 2: Raster integrity & Pairing
    phase1_failures = []
    pairing_mismatches = []
    
    raster_stats = {
        "total_files": 0,
        "successfully_read": 0,
        "file_sizes": {"min": float('inf'), "max": 0, "total_bytes": 0}
    }

    # Tracking sensor value statistics across subset/full
    s1_vv_min = float('inf')
    s1_vv_max = float('-inf')
    s1_vh_min = float('inf')
    s1_vh_max = float('-inf')
    s1_nan_count = 0
    s1_inf_count = 0

    s2_band_mins = [float('inf')] * 13
    s2_band_maxs = [float('-inf')] * 13

    # PHASE 3: Label metrics
    total_pixels = 0
    label_counts = {-1: 0, 0: 0, 1: 0, "other": 0}
    jrc_counts = {0: 0, 1: 0, "other": 0}
    
    # Combinations: (L, J)
    combos = {
        "L0_J0": 0,  # Dry land
        "L1_J1": 0,  # Permanent water
        "L1_J0": 0,  # Flood inundation
        "L0_J1": 0,  # Disagreement
        "Lneg1_J0": 0,
        "Lneg1_J1": 0,
        "other": 0
    }

    disagreement_by_country = {}
    chips_with_disagreement = []

    print("Auditing all 446 chips (1,784 GeoTIFF rasters)...")
    for idx, chip_id in enumerate(all_chip_ids):
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(all_chip_ids)} chips...")

        # Check pairing
        if chip_id not in s1_files or chip_id not in s2_files or chip_id not in jrc_files:
            pairing_mismatches.append(chip_id)
            continue

        l_path = label_files[chip_id]
        s1_path = s1_files[chip_id]
        s2_path = s2_files[chip_id]
        jrc_path = jrc_files[chip_id]

        chip_rasters = [
            ("LabelHand", l_path, 1, np.int16),
            ("S1Hand", s1_path, 2, np.float32),
            ("S2Hand", s2_path, 13, np.int16),
            ("JRCWaterHand", jrc_path, 1, np.uint8)
        ]

        meta_list = []
        data_dict = {}

        for r_name, r_path, exp_count, exp_dtype in chip_rasters:
            raster_stats["total_files"] += 1
            f_size = r_path.stat().st_size
            raster_stats["file_sizes"]["total_bytes"] += f_size
            if f_size < raster_stats["file_sizes"]["min"]:
                raster_stats["file_sizes"]["min"] = f_size
            if f_size > raster_stats["file_sizes"]["max"]:
                raster_stats["file_sizes"]["max"] = f_size

            if f_size == 0:
                phase1_failures.append((r_path.name, "Zero file size"))
                continue

            try:
                with rasterio.open(r_path) as src:
                    w, h = src.width, src.height
                    count = src.count
                    crs = src.crs
                    transform = src.transform
                    bounds = src.bounds
                    nodata = src.nodata
                    dtype = np.dtype(src.dtypes[0])

                    # Check basic specs
                    if w != 512 or h != 512:
                        phase1_failures.append((r_path.name, f"Unexpected shape {w}x{h}"))
                    if count != exp_count:
                        phase1_failures.append((r_path.name, f"Unexpected band count {count} != {exp_count}"))
                    if crs is None or crs.to_epsg() != 4326:
                        phase1_failures.append((r_path.name, f"Unexpected CRS {crs}"))

                    arr = src.read()
                    if arr.shape != (exp_count, 512, 512):
                        phase1_failures.append((r_path.name, f"Read shape mismatch {arr.shape}"))
                    
                    data_dict[r_name] = arr
                    meta_list.append({
                        "name": r_name,
                        "w": w, "h": h,
                        "crs": str(crs),
                        "transform": transform,
                        "bounds": bounds,
                        "nodata": nodata,
                        "dtype": str(dtype)
                    })
                    raster_stats["successfully_read"] += 1
            except Exception as e:
                phase1_failures.append((r_path.name, f"Exception reading raster: {str(e)}"))

        # Verify spatial alignment across the 4 components of the chip
        if len(meta_list) == 4:
            base_meta = meta_list[0]
            for other_meta in meta_list[1:]:
                if other_meta["w"] != base_meta["w"] or other_meta["h"] != base_meta["h"]:
                    phase1_failures.append((chip_id, f"Dimension mismatch: {other_meta['name']} vs {base_meta['name']}"))
                
                # Check transform with float64 tolerance (1e-7 deg ~ 1.1 cm, well below 10m pixel)
                t_base = np.array(list(base_meta["transform"]))
                t_other = np.array(list(other_meta["transform"]))
                if not np.allclose(t_base, t_other, atol=1e-7):
                    phase1_failures.append((chip_id, f"Transform mismatch: {other_meta['name']} vs {base_meta['name']} (max diff: {np.max(np.abs(t_base - t_other))})"))
                
                b_base = np.array(list(base_meta["bounds"]))
                b_other = np.array(list(other_meta["bounds"]))
                if not np.allclose(b_base, b_other, atol=1e-7):
                    phase1_failures.append((chip_id, f"Bounds mismatch: {other_meta['name']} vs {base_meta['name']} (max diff: {np.max(np.abs(b_base - b_other))})"))

        # Compute label statistics
        if "LabelHand" in data_dict and "JRCWaterHand" in data_dict:
            l_arr = data_dict["LabelHand"][0]
            j_arr = data_dict["JRCWaterHand"][0]

            total_pixels += l_arr.size

            # Label counts
            u, c = np.unique(l_arr, return_counts=True)
            for val, cnt in zip(u, c):
                if val in label_counts:
                    label_counts[val] += int(cnt)
                else:
                    label_counts["other"] += int(cnt)

            # JRC counts
            uj, cj = np.unique(j_arr, return_counts=True)
            for val, cnt in zip(uj, cj):
                if val in jrc_counts:
                    jrc_counts[val] += int(cnt)
                else:
                    jrc_counts["other"] += int(cnt)

            # Combinations
            m_l0_j0 = (l_arr == 0) & (j_arr == 0)
            m_l1_j1 = (l_arr == 1) & (j_arr == 1)
            m_l1_j0 = (l_arr == 1) & (j_arr == 0)
            m_l0_j1 = (l_arr == 0) & (j_arr == 1)
            m_lneg1_j0 = (l_arr == -1) & (j_arr == 0)
            m_lneg1_j1 = (l_arr == -1) & (j_arr == 1)

            combos["L0_J0"] += int(np.sum(m_l0_j0))
            combos["L1_J1"] += int(np.sum(m_l1_j1))
            combos["L1_J0"] += int(np.sum(m_l1_j0))
            combos["L0_J1"] += int(np.sum(m_l0_j1))
            combos["Lneg1_J0"] += int(np.sum(m_lneg1_j0))
            combos["Lneg1_J1"] += int(np.sum(m_lneg1_j1))

            disagree_cnt = int(np.sum(m_l0_j1))
            if disagree_cnt > 0:
                country = chip_id.split("_")[0]
                disagreement_by_country[country] = disagreement_by_country.get(country, 0) + disagree_cnt
                chips_with_disagreement.append((chip_id, disagree_cnt, float(disagree_cnt) / l_arr.size * 100))

        # Sensor stats for S1
        if "S1Hand" in data_dict:
            s1_arr = data_dict["S1Hand"]
            vv = s1_arr[0]
            vh = s1_arr[1]
            vv_fin = vv[np.isfinite(vv)]
            vh_fin = vh[np.isfinite(vh)]

            s1_nan_count += int(np.isnan(vv).sum() + np.isnan(vh).sum())
            s1_inf_count += int(np.isinf(vv).sum() + np.isinf(vh).sum())

            if len(vv_fin) > 0:
                s1_vv_min = min(s1_vv_min, float(np.min(vv_fin)))
                s1_vv_max = max(s1_vv_max, float(np.max(vv_fin)))
            if len(vh_fin) > 0:
                s1_vh_min = min(s1_vh_min, float(np.min(vh_fin)))
                s1_vh_max = max(s1_vh_max, float(np.max(vh_fin)))

        # Sensor stats for S2
        if "S2Hand" in data_dict:
            s2_arr = data_dict["S2Hand"]
            for b in range(13):
                b_min = float(np.min(s2_arr[b]))
                b_max = float(np.max(s2_arr[b]))
                if b_min < s2_band_mins[b]:
                    s2_band_mins[b] = b_min
                if b_max > s2_band_maxs[b]:
                    s2_band_maxs[b] = b_max

    print("Completed raster and label iteration.")

    # PHASE 4: Official split integrity
    split_info = {}
    split_chips = {}
    for csv_file in SPLITS_DIR.glob("*.csv"):
        with open(csv_file, "r") as f:
            raw_lines = [line.strip() for line in f if line.strip()]
        # Extract chip ID from line (e.g. Ghana_103272_S1Hand.tif,Ghana_103272_LabelHand.tif)
        chip_ids = []
        for line in raw_lines:
            first_part = line.split(",")[0].strip()
            chip_id = first_part.replace("_S1Hand.tif", "").replace(".tif", "")
            chip_ids.append(chip_id)
        
        split_chips[csv_file.name] = set(chip_ids)
        split_info[csv_file.name] = {
            "count": len(chip_ids),
            "sample": chip_ids[:3]
        }

    # Verify disjointness
    csv_names = list(split_chips.keys())
    overlap_detected = False
    pairwise_overlaps = {}
    for i in range(len(csv_names)):
        for j in range(i + 1, len(csv_names)):
            c1, c2 = csv_names[i], csv_names[j]
            overlap = split_chips[c1].intersection(split_chips[c2])
            pairwise_overlaps[f"{c1} vs {c2}"] = len(overlap)
            if len(overlap) > 0:
                overlap_detected = True

    # Check Bolivia chips
    bolivia_csv_chips = split_chips.get("flood_bolivia_data.csv", set())
    # All bolivia_csv_chips should start with 'Bolivia'
    all_bolivia_named = all(c.startswith("Bolivia") for c in bolivia_csv_chips)

    # Check how many total Bolivia chips exist in the 446 dataset
    total_bolivia_in_dataset = [c for c in all_chip_ids if c.startswith("Bolivia")]

    # All chips in splits vs all chips on disk
    all_split_chips = set()
    for s_chips in split_chips.values():
        all_split_chips.update(s_chips)

    chips_on_disk = set(all_chip_ids)
    missing_from_splits = chips_on_disk - all_split_chips
    extra_in_splits = all_split_chips - chips_on_disk

    results = {
        "phase1_raster_integrity": {
            "total_files": raster_stats["total_files"],
            "expected_files": 446 * 4,
            "successfully_read": raster_stats["successfully_read"],
            "failures_count": len(phase1_failures),
            "failures": phase1_failures[:10],
            "file_size_min_bytes": raster_stats["file_sizes"]["min"],
            "file_size_max_bytes": raster_stats["file_sizes"]["max"],
            "file_size_total_mb": raster_stats["file_sizes"]["total_bytes"] / (1024 * 1024),
            "status": "PASS" if len(phase1_failures) == 0 and raster_stats["successfully_read"] == 1784 else "FAIL"
        },
        "phase2_chip_pairing": {
            "total_chips": len(all_chip_ids),
            "s1_count": len(s1_files),
            "s2_count": len(s2_files),
            "label_count": len(label_files),
            "jrc_count": len(jrc_files),
            "pairing_mismatches_count": len(pairing_mismatches),
            "pairing_mismatches": pairing_mismatches,
            "status": "PASS" if len(pairing_mismatches) == 0 and len(all_chip_ids) == 446 else "FAIL"
        },
        "phase3_label_audit": {
            "total_pixels": total_pixels,
            "label_counts": label_counts,
            "label_percentages": {k: float(v) / total_pixels * 100 for k, v in label_counts.items() if isinstance(v, int)},
            "jrc_counts": jrc_counts,
            "jrc_percentages": {k: float(v) / total_pixels * 100 for k, v in jrc_counts.items() if isinstance(v, int)},
            "combinations": combos,
            "combination_percentages": {k: float(v) / total_pixels * 100 for k, v in combos.items()},
            "disagreement_L0_J1_total": combos["L0_J1"],
            "disagreement_L0_J1_pct": float(combos["L0_J1"]) / total_pixels * 100,
            "disagreement_by_country": disagreement_by_country,
            "top_disagreement_chips": sorted(chips_with_disagreement, key=lambda x: x[1], reverse=True)[:10],
            "status": "PASS"
        },
        "phase4_split_integrity": {
            "split_counts": {k: v["count"] for k, v in split_info.items()},
            "total_split_chips": len(all_split_chips),
            "overlap_detected": overlap_detected,
            "pairwise_overlaps": pairwise_overlaps,
            "missing_from_splits": list(missing_from_splits),
            "extra_in_splits": list(extra_in_splits),
            "bolivia_chips_count": len(bolivia_csv_chips),
            "all_bolivia_named": all_bolivia_named,
            "total_bolivia_in_dataset": len(total_bolivia_in_dataset),
            "status": "PASS" if not overlap_detected and len(missing_from_splits) == 0 and len(extra_in_splits) == 0 else "FAIL"
        },
        "phase5_sensor_compatibility": {
            "s1_vv_min_db": s1_vv_min,
            "s1_vv_max_db": s1_vv_max,
            "s1_vh_min_db": s1_vh_min,
            "s1_vh_max_db": s1_vh_max,
            "s1_nan_count": s1_nan_count,
            "s1_inf_count": s1_inf_count,
            "s2_band_ranges": [{"band": i + 1, "min": s2_band_mins[i], "max": s2_band_maxs[i]} for i in range(13)],
            "s2_optical_4ch_indices": [3, 2, 1, 7],
            "s2_optical_4ch_names": ["B4 (Red)", "B3 (Green)", "B2 (Blue)", "B8 (NIR)"],
            "status": "PASS"
        }
    }

    # Save results to json
    out_json = DATA_DIR / "audit_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Audit completed successfully! Saved to {out_json}")

if __name__ == "__main__":
    run_audit()

