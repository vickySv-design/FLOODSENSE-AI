# FloodSense-AI: Comprehensive Scenario Default Values Reference

**Document Reference:** `SCENARIO_DEFAULT_VALUES.md`  
**Applicability:** FloodSense-AI 2D Hydrodynamic & Cyclone Scenario Simulation Engine  
**Standardized Reference System:** WMO / IMD Climatological Guidelines & USGS Hydraulic Engineering Standards  

---

## 1. Overview of Effective Parameters

Whenever a user executes a disaster simulation without specifying certain operational parameters, FloodSense-AI resolves each missing field using scientifically vetted, documented defaults. Under no circumstances are values substituted silently.

Each parameter is cataloged with its **Physical Units**, **Baseline Default Value**, **Provenance Classification**, and **Scientific & Physical Justification**.

---

## 2. Exhaustive 14-Parameter Reference Matrix

| Parameter # | Parameter Name | Canonical Key | Default Value | Physical Units | Provenance Source | Scientific / Climatological Justification |
|:---:|---|---|:---:|:---:|:---:|---|
| **1** | **Rainfall Intensity** | `rainfall_intensity_mm_hr` | `45.0` | mm/hour | `BACKEND_DEFAULT` | Matches the Indian Meteorological Department (IMD) standard threshold for a "Severe Monsoon Depressive Burst" or heavy rainband passage ($40-60\text{ mm/hr}$). |
| **2** | **Maximum Sustained Wind Speed** | `wind_speed_kmh` | `120.0` | km/hour | `BACKEND_DEFAULT` | Standard IMD reference speed for a "Very Severe Cyclonic Storm" (VSCS, threshold $117-165\text{ km/h}$), representative of Bay of Bengal post-monsoon systems (e.g. Cyclone Vardah, Cyclone Michaung). |
| **3** | **Rainfall & Storm Track Direction** | `rainfall_direction_deg` | `315.0` | degrees ($0-360^\circ$) | `BACKEND_DEFAULT` | Represents a North-West ($315^\circ$) landfall track vector, the dominant climatological trajectory for tropical cyclones tracking across the North Indian Ocean / Bay of Bengal basin toward peninsular India. |
| **4** | **Cyclone Movement Speed** | `movement_speed_kmh` | `18.0` | km/hour | `BACKEND_DEFAULT` | Climatological median translational forward speed ($15-20\text{ km/h}$) observed across North Indian Ocean tropical depressions making coastal transit. |
| **5** | **Cyclone Centre Coordinates** | `cyclone_centre_coordinates` | Geometric AOI Centroid | Latitude, Longitude | `BACKEND_DEFAULT` | Calculated dynamically as `[0.5*(min_lat + max_lat), 0.5*(min_lon + max_lon)]`. Maximizes initial storm interaction with the operator-selected study area. |
| **6** | **Radius of Maximum Winds (RMW)** | `radius_maximum_winds_km` | `45.0` | km | `BACKEND_DEFAULT` | Calibrated based on Holland vortex wind profile empirical studies for Indian Ocean cyclones, corresponding to the radial distance between the eye center and the peak eyewall convective ring. |
| **7** | **Simulation Duration Horizon** | `simulation_duration_hours` | `24` | hours | `BACKEND_DEFAULT` | Standard emergency operations briefing cycle (24-hour tactical horizon) established by National Disaster Response Force (NDRF) Standard Operating Procedures. |
| **8** | **Numerical Simulation Timestep** | `simulation_timestep_sec` | `180.0` | seconds ($3\text{ min}$) | `BACKEND_DEFAULT` | Satisfies the 2D Courant-Friedrichs-Lewy (CFL) numerical stability criterion: $\Delta t \le \frac{\Delta x}{\sqrt{g \cdot h_{\max}}}$ for a $10\text{m}$ grid cell and typical overland depths ($<2.5\text{m}$). |
| **9** | **Hyetograph Distribution Model** | `rainfall_distribution_model` | `front_loaded` | dimensionless profile curve | `BACKEND_DEFAULT` | Convective cloudburst profile: precipitation surges rapidly to peak intensity within the first quartile ($0.25 \times T_{\text{dur}}$), simulating sudden urban drainage overwhelming before tapering exponentially. |
| **10** | **Terrain Topography Source** | `terrain_source` | Hydro-conditioned DEM / Procedural Talweg | Elevation matrix | `SYNTHETIC_ASSUMPTION` / `REAL_EXTERNAL_DATA` | If pre-calibrated Copernicus GLO-30 DEM exists on disk for preset, uses `REAL_EXTERNAL_DATA`. Otherwise, synthesizes a hydro-conditioned fractal talweg with natural regional slope gradients (`SYNTHETIC_ASSUMPTION`). |
| **11** | **DEM Spatial Resolution** | `dem_source_and_resolution` | `10.0` | meters per pixel ($256 \times 256$ grid) | `BACKEND_DEFAULT` / `REAL_EXTERNAL_DATA` | Strictly calibrated to match European Space Agency (ESA) Sentinel-1 C-SAR and Sentinel-2 MSI 10-meter Ground Sampling Distance (GSD) for 1:1 geospatial overlay accuracy. |
| **12** | **Soil Infiltration Capacity** | `infiltration_parameters` | `8.0` | mm/hour ($2.22 \times 10^{-6}\text{ m/s}$) | `SYNTHETIC_ASSUMPTION` | Represents ultimate saturated hydraulic conductivity ($K_{\text{sat}}$) based on the Horton infiltration model for semi-urban clay loam soil under antecedent saturated conditions. |
| **13** | **Municipal Drainage Capacity** | `drainage_parameters` | `14.0` | mm/hour ($3.89 \times 10^{-6}\text{ m/s}$) | `SYNTHETIC_ASSUMPTION` | Standard empirical urban stormwater carrying capacity representing roadside drains, culverts, and canal outfalls before gravity drainage is inhibited by downstream backwater. |
| **14** | **Initial Surface Water Depth** | `initial_water_depth_m` | `0.0` | meters | `SYNTHETIC_ASSUMPTION` | Initial condition assuming dry antecedent ground surface prior to event initiation; baseline permanent water bodies are maintained separately via the land-cover mask. |

---

## 3. Provenance Classification Taxonomy

FloodSense-AI categorizes every parameter into one of five unambiguous provenance levels:

1. `USER_INPUT`: Value was explicitly typed or adjusted by the human operator in the UI or API request.
2. `SELECTED_PRESET`: Value was inherited from a vetted, pre-packaged historical disaster scenario (e.g. *Chennai 2026*, *Michaung Surge*).
3. `BACKEND_DEFAULT`: Parameter was unassigned by user and was populated with an officially documented meteorological/hydrological standard.
4. `REAL_EXTERNAL_DATA`: Value or raster grid was fetched from an authenticated, genuine external telemetry provider (Copernicus DEM, GEE, Open-Meteo).
5. `SYNTHETIC_ASSUMPTION`: Parameter represents a necessary numerical or physical assumption (e.g. uniform infiltration, closed boundary) due to absence of local telemetry.

---

## 4. Dual-Scale Cyclone Wind Categorization Standards

To ensure cross-jurisdictional compliance during operations, wind speeds are simultaneously classified according to two meteorological scales:

| Maximum Sustained Wind ($V_{3\text{min}}$) | IMD Scale (3-minute sustained) | Equivalent 1-min Wind ($V_{1\text{min}} = 1.14 \times V_{3\text{min}}$) | Saffir-Simpson Hurricane Wind Scale (1-minute sustained) |
|---|---|---|---|
| $< 51\text{ km/h}$ | **Depression** ($31-50\text{ km/h}$) | $< 58\text{ km/h}$ | **Tropical Depression** ($< 63\text{ km/h}$) |
| $51 - 61\text{ km/h}$ | **Deep Depression** ($51-61\text{ km/h}$) | $58 - 70\text{ km/h}$ | **Tropical Depression / Weak Storm** |
| $62 - 87\text{ km/h}$ | **Cyclonic Storm** ($62-87\text{ km/h}$) | $71 - 99\text{ km/h}$ | **Tropical Storm** ($63-118\text{ km/h}$) |
| $88 - 116\text{ km/h}$ | **Severe Cyclonic Storm** ($88-116\text{ km/h}$) | $100 - 132\text{ km/h}$ | **Severe Storm / Cat 1 Hurricane** ($119-153\text{ km/h}$) |
| $117 - 165\text{ km/h}$ | **Very Severe Cyclonic Storm (VSCS)** | $133 - 188\text{ km/h}$ | **Category 1 to Category 2 Hurricane** |
| $166 - 220\text{ km/h}$ | **Extremely Severe Cyclonic Storm (ESCS)** | $189 - 251\text{ km/h}$ | **Category 3 to Category 4 Major Hurricane** |
| $\ge 221\text{ km/h}$ | **Super Cyclonic Storm** ($\ge 221\text{ km/h}$) | $\ge 252\text{ km/h}$ | **Category 5 Major Hurricane** ($\ge 252\text{ km/h}$) |

> [!NOTE]
> The conversion factor $V_{1\text{min}} = 1.14 \times V_{3\text{min}}$ is the official standard adopted by the World Meteorological Organization (WMO) for converting between 3-minute sustained winds (IMD, BoM) and 1-minute sustained winds (NOAA National Hurricane Center).

