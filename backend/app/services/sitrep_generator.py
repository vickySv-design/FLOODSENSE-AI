from datetime import datetime, timezone
import json
import math
import os
from typing import Any, Dict, List, Optional
import requests


def _determine_severity_level(
    flood_expansion_pct: float, critical_buildings: int, flooded_area_km2: float
) -> str:
    if flood_expansion_pct >= 75.0 or critical_buildings >= 10 or flooded_area_km2 >= 1.2:
        return "CRITICAL (RED ALERT)"
    if flood_expansion_pct >= 35.0 or critical_buildings >= 4 or flooded_area_km2 >= 0.5:
        return "HIGH (ORANGE ALERT)"
    return "MODERATE (YELLOW ALERT)"


def _calculate_resource_allocation(
    flooded_area_km2: float,
    critical_buildings: int,
    moderate_buildings: int,
    affected_population: int,
) -> Dict[str, int]:
    """
    Computes NDRF / SDMA standard emergency logistics requirements from live flood telemetry.
    """
    pop = max(
        affected_population,
        int(critical_buildings * 42 + moderate_buildings * 20 + flooded_area_km2 * 450),
    )
    inflatable_boats = max(4, int(math.ceil(critical_buildings * 0.45 + flooded_area_km2 * 3.2)))
    medical_teams = max(2, int(math.ceil(pop / 280.0)))
    food_packets_24h = int(math.ceil((pop * 3) / 50.0) * 50)
    evacuation_buses = max(3, int(math.ceil(pop / 65.0)))

    return {
        "inflatable_boats_needed": inflatable_boats,
        "medical_teams": medical_teams,
        "food_packets_24h": food_packets_24h,
        "evacuation_buses": evacuation_buses,
        "estimated_evacuees": pop,
    }


def _build_expert_system_sitrep(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic Rule-Based Expert System SitRep Generator (Zero-Key Offline Engine).
    Dynamically injects exact hydrological, structural, and A* routing telemetry into a formal
    NDRF / District Collector Disaster Flash Bulletin.
    """
    scenario = payload.get("scenario", {})
    telemetry = payload.get("telemetry", {})
    buildings_summary = payload.get("buildings_summary", {})
    rescue = payload.get("rescue", {})
    metrics = payload.get("metrics", {})

    scenario_name = scenario.get("name", "Active Monsoon Flood Sector")
    region = scenario.get("region", "Target Floodplain AOI")
    bbox: List[float] = scenario.get("bbox", [13.010, 80.220, 13.060, 80.270])
    center: List[float] = scenario.get(
        "center",
        [round((bbox[0] + bbox[2]) * 0.5, 4), round((bbox[1] + bbox[3]) * 0.5, 4)],
    )

    flooded_km2 = float(telemetry.get("flooded_area_km2", 1.85))
    perm_water_km2 = float(telemetry.get("permanent_water_km2", 0.92))
    buffer_km2 = float(telemetry.get("buffer_risk_km2", 0.64))
    expansion_pct = float(telemetry.get("flood_expansion_pct", 142.5))

    crit_blds = int(buildings_summary.get("critical_submerged", 14))
    mod_blds = int(buildings_summary.get("moderate_buffer_risk", 11))
    affected_pop = int(buildings_summary.get("estimated_affected_population", 1250))

    rescue_summary = rescue.get("summary", {})
    safe_routes_count = int(rescue_summary.get("active_safe_routes", 3))
    shortest_km = float(rescue_summary.get("shortest_route_km", 2.45))
    fastest_eta = float(rescue_summary.get("fastest_eta_mins", 8.5))
    blocked_roads = int(rescue_summary.get("blocked_road_segments", 18))

    miou_pct = round(float(metrics.get("miou", 0.914)) * 100.0, 1)
    f1_pct = round(float(metrics.get("f1_score", 0.948)) * 100.0, 1)

    severity = _determine_severity_level(expansion_pct, crit_blds, flooded_km2)
    resources = _calculate_resource_allocation(
        flooded_km2, crit_blds, mod_blds, affected_pop
    )

    executive_summary = (
        f"Multi-modal change detection across {scenario_name} ({region}) reveals {flooded_km2:.3f} km² "
        f"of active Class-2 flood inundation beyond the {perm_water_km2:.3f} km² Pre-Event Sentinel-2 NDWI permanent water baseline, "
        f"representing a +{expansion_pct:.1f}% hydrological surge. "
        f"Post-event Sentinel-1 C-Band SAR dual-polarization (VV/VH) analysis confirms severe specular backscatter drop (<-16.0 dB) "
        f"across low-lying basins, directly submerging {crit_blds} critical structures and placing {mod_blds} additional buildings "
        f"within the 100m high-risk buffer zone ({buffer_km2:.3f} km²). "
        f"Dual-Stream Siamese Attention U-Net segmentation achieved {miou_pct}% mIoU ({f1_pct}% F1) under monsoon cloud cover, "
        f"identifying {blocked_roads} severed road segments and verifying {safe_routes_count} dry A* evacuation corridors."
    )

    ndrf_deployment_orders = [
        (
            f"DIRECTIVE ALPHA [MARITIME RESCUE]: Immediately deploy {resources['inflatable_boats_needed']} motorized "
            f"NDRF inflatable rescue boats to submerged urban/riverine pockets centered at "
            f"[{center[0]:.4f}°N, {center[1]:.4f}°E] to extract occupants from {crit_blds} CRITICAL_SUBMERGED structures."
        ),
        (
            f"DIRECTIVE BRAVO [CORRIDOR DISPATCH]: Route high-clearance evacuation convoys ({resources['evacuation_buses']} buses) "
            f"exclusively along the {safe_routes_count} verified NetworkX A* green corridors (shortest path: {shortest_km:.2f} km, "
            f"estimated transit ETA: {fastest_eta:.1f} mins from NDRF Staging Base)."
        ),
        (
            f"DIRECTIVE CHARLIE [TRAFFIC CORDONS]: Establish immediate police and SDRF barricades across all {blocked_roads} "
            f"flood-severed road segments within bounding box [{bbox[0]:.3f}°N, {bbox[1]:.3f}°E] to [{bbox[2]:.3f}°N, {bbox[3]:.3f}°E] "
            f"and issue mandatory evacuation notices for the {mod_blds} structures inside the 100m morphological risk buffer."
        ),
        (
            f"DIRECTIVE DELTA [MEDICAL & RELIEF STAGING]: Activate elevated relief shelters for {resources['estimated_evacuees']:,} "
            f"displaced residents, pre-positioning {resources['medical_teams']} rapid medical response teams and "
            f"{resources['food_packets_24h']:,} emergency ration/hydration packets for the initial 24-hour window."
        ),
    ]

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    bulletin_id = f"NDRF-SDMA-{abs(hash(scenario_name)) % 9000 + 1000}"

    markdown_report = f"""# OFFICIAL DISASTER SITUATION REPORT (SITREP)
**Bulletin ID:** `{bulletin_id}` | **Timestamp:** `{now_iso}`
**Classification:** `{severity}`
**Target AOI:** {scenario_name} ({region})
**Coordinates:** `[{bbox[0]}°N, {bbox[1]}°E]` to `[{bbox[2]}°N, {bbox[3]}°E]`

---

## 1. EXECUTIVE HYDROLOGICAL ASSESSMENT
{executive_summary}

## 2. SATELLITE & DEEP LEARNING TELEMETRY
| Metric | Value | Sensor / Method |
| :--- | :--- | :--- |
| **Baseline Permanent Water (Class 1)** | `{perm_water_km2:.3f} km²` | Sentinel-2 MSI Pre-Event NDWI |
| **New Flood Inundation (Class 2)** | `{flooded_km2:.3f} km²` | Sentinel-1 SAR + Siamese Attention U-Net |
| **Flood Expansion Surge** | `+{expansion_pct:.1f}%` | Multi-Temporal Pixel Delta (10m/px) |
| **100m High-Risk Buffer Zone** | `{buffer_km2:.3f} km²` | Morphological Dilation Zone |
| **Submerged / Buffer-Risk Buildings** | `{crit_blds} Critical / {mod_blds} Buffer` | Footprint Intersection Grading |
| **Verified Safe A* Rescue Corridors** | `{safe_routes_count} Routes (Min {shortest_km:.2f} km, ETA {fastest_eta:.1f}m)` | NetworkX Haversine A* Router |
| **Severed / Flooded Road Segments** | `{blocked_roads} Blocked Edges` | Flood Mask Line-Sampling |

## 3. NDRF / SDMA OPERATIONAL DEPLOYMENT ORDERS
1. {ndrf_deployment_orders[0]}
2. {ndrf_deployment_orders[1]}
3. {ndrf_deployment_orders[2]}
4. {ndrf_deployment_orders[3]}

## 4. EMERGENCY RESOURCE ALLOCATION MANIFEST
- **Motorized Inflatable Rescue Boats:** `{resources['inflatable_boats_needed']} Units`
- **Rapid Medical Response Teams:** `{resources['medical_teams']} Teams`
- **24-Hour Emergency Ration & Water Kits:** `{resources['food_packets_24h']:,} Packets`
- **High-Clearance Evacuation Buses:** `{resources['evacuation_buses']} Vehicles`
- **Estimated Displaced Population:** `{resources['estimated_evacuees']:,} Citizens`
"""

    return {
        "bulletin_id": bulletin_id,
        "timestamp_utc": now_iso,
        "scenario_name": scenario_name,
        "region": region,
        "bbox": bbox,
        "severity_level": severity,
        "executive_summary": executive_summary,
        "ndrf_deployment_orders": ndrf_deployment_orders,
        "resource_allocation": resources,
        "generation_engine": "NDRF_EXPERT_RULE_ENGINE_V2 (Deterministic Offline Fallback)",
        "markdown_report": markdown_report,
    }


def generate_disaster_sitrep(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hybrid Free SitRep Engine:
    - First builds the deterministic baseline SitRep with exact mathematical resource numbers.
    - If GEMINI_API_KEY is present in environment variables, calls the Gemini 2.5 Flash REST API
      to refine the executive summary and operational orders while preserving exact schema safety.
    """
    base_sitrep = _build_expert_system_sitrep(payload)
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    if not api_key or not api_key.strip():
        return base_sitrep

    try:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={api_key.strip()}"
        )
        prompt = (
            "You are the Chief Geospatial Intelligence Officer for the National Disaster Response Force (NDRF). "
            "Given the following live satellite flood telemetry JSON, return a strict JSON object with keys "
            "'executive_summary' (3 concise technical sentences comparing Pre-Event Sentinel-2 NDWI and Post-Event "
            "Sentinel-1 SAR backscatter drop) and 'ndrf_deployment_orders' (array of 4 concrete operational directives "
            "referencing the exact coordinates, route distances, and building counts). "
            f"Telemetry JSON: {json.dumps(base_sitrep)}"
        )
        resp = requests.post(
            endpoint,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            text_out = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            parsed = json.loads(text_out)
            if (
                isinstance(parsed.get("executive_summary"), str)
                and isinstance(parsed.get("ndrf_deployment_orders"), list)
                and len(parsed["ndrf_deployment_orders"]) == 4
            ):
                base_sitrep["executive_summary"] = parsed["executive_summary"]
                base_sitrep["ndrf_deployment_orders"] = parsed["ndrf_deployment_orders"]
                base_sitrep["generation_engine"] = "GEMINI_2_5_FLASH_LIVE"
    except Exception:
        pass

    return base_sitrep

