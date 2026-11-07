# FloodSense-AI — UI Redesign Feature Mapping & Architecture Plan

## 1. Audit of Existing Application Architecture

### 1.1 Existing Component Inventory & Functionality Status

| Existing Component | File Path | Current Role / UI Form | Verified Functionality Status | Discovered Limitations / Notes |
|---|---|---|---|---|
| **App.jsx** | `frontend/src/App.jsx` | Monolithic root holding all modal states, analysis data, and viewport toggles | Functional | All views crammed into single page; heavy state mixing. |
| **HeaderBar.jsx** | `frontend/src/components/HeaderBar.jsx` | Dark HUD topbar with telemetry badges, hardware stats, and 6 modal launcher buttons | Functional | Dark cyberpunk HUD theme; crowded button layout. |
| **ControlSidebar.jsx** | `frontend/src/components/ControlSidebar.jsx` | Left dark drawer with 3 tabs (Presets, Upload, GEE) + Layer toggles | Functional | Dense military styling; mixes AOI selection, upload, and layer visibility. |
| **MapViewport.jsx** | `frontend/src/components/MapViewport.jsx` | Central Leaflet map container with ESRI, Dark, OSM tiles, GeoJSON overlays | Functional | High-performance Leaflet; dark-themed popups; excellent layer rendering. |
| **SplitCompareSlider.jsx** | `frontend/src/components/SplitCompareSlider.jsx` | Split-screen curtain comparing Pre Optical vs Post SAR / NDWI / Attention | Functional | Responsive drag handle; side-by-side raster comparison. |
| **AnalyticsDock.jsx** | `frontend/src/components/AnalyticsDock.jsx` | Bottom drawer with 4 KPI cards and 3 subtabs (Hydrology, Timeline, Benchmarks) | Functional | Deep charts and tables, but pinned to the bottom causing scrolling friction. |
| **TimelineSimulatorPanel.jsx**| `frontend/src/components/TimelineSimulatorPanel.jsx` | 48-Hour rainfall intensity slider and flood surge/recession area chart | Functional | Connects to `/api/simulate-timeline`. |
| **ModelBenchmarkTab.jsx** | `frontend/src/components/ModelBenchmarkTab.jsx` | Architecture ablation chart comparing U-Net, ResNet, and Attention U-Net | Functional | Connects to `/api/benchmarks` for viva defense. |
| **ProjectWalkthroughModal.jsx** | `frontend/src/components/ProjectWalkthroughModal.jsx` | Multi-step interactive guided project defense modal | Functional | Complete slide deck and architecture explanation for viva. |
| **SitRepModal.jsx** | `frontend/src/components/SitRepModal.jsx` | AI NDRF Situation Report modal with markdown viewer and print trigger | Functional | Connects to `/api/generate-sitrep`, exports markdown and GeoJSON. |
| **UniversalAOIModal.jsx** | `frontend/src/components/UniversalAOIModal.jsx` | Modal with Nominatim search, custom bounding box, and GeoJSON upload | Functional | Connects to `/api/aoi/search`, `/api/aoi/validate`, and dynamic AOI analysis. |
| **DataAcquisitionModal.jsx** | `frontend/src/components/DataAcquisitionModal.jsx` | Multi-tab modal for Copernicus, DEM, OSM, Open-Meteo, NASA GPM, Cache | Functional / Partially Functional | Real OSM & Open-Meteo; Copernicus searches are real; DEM is synthetic fractal. |
| **ScenarioSimulationModal.jsx**| `frontend/src/components/ScenarioSimulationModal.jsx` | Full 2D hydrodynamic cyclone simulator lab with 14 parameters and timeline | Functional | Dynamic diffusive wave simulation, real OSM sampling, mass balance error. |

---

## 2. Information Architecture Reorganization

The monolithic single-page layout is reorganized into an **Application Shell with Collapsible Sidebar Navigation** and **7 dedicated Pages** styled in a modern professional light theme.

```
+---------------------------------------------------------------------------------------+
|  TOP NAVIGATION BAR                                                                   |
|  [Logo & FloodSense-AI] | Breadcrumbs | Active AOI Badge | System Status | Quick Viva |
+-----------------------+---------------------------------------------------------------+
| SIDEBAR NAVIGATION    | MAIN CONTENT CONTAINER (Modern Professional Light Theme)       |
|                       |                                                               |
| [x] Dashboard         | Page 1: Disaster Monitoring Overview                          |
| [ ] Satellite Analysis| Page 2: Multi-Modal Satellite Analysis Pipeline               |
| [ ] Flood Simulation  | Page 3: 2D Hydrodynamic & Cyclone Simulation Lab              |
| [ ] Rescue & Infra    | Page 4: Emergency Rescue Routing & Infrastructure Exposure    |
| [ ] Disaster Reports  | Page 5: NDRF Situation Reports & Multi-Format Exports          |
| [ ] Data Sources      | Page 6: Live API Providers & Offline Cache Manager            |
| [ ] Settings          | Page 7: Application Configuration & System Diagnostics        |
+-----------------------+---------------------------------------------------------------+
```

---

## 3. Detailed Feature-to-Page Mapping

### Page 1: Dashboard (`#dashboard`)
- **Primary Purpose:** Executive overview of disaster status, impact metrics, and spatial overview.
- **Header:** "Disaster Monitoring Overview" with subtitle "Monitor flood conditions, assess infrastructure exposure and explore emergency response information."
- **Summary Cards:**
  - Newly Inundated Area ($km^2$)
  - Buildings Exposed (Count & Affected Population)
  - Severed / Blocked Roads (Count & Length)
  - Candidate Safe Rescue Corridors (Count & Fastest ETA)
- **Interactive GIS Map:** Large interactive Leaflet map with ESRI satellite, Carto, and OSM basemaps, full layer toggles (all roads, flood polygons, permanent water, risk buffer, building footprints, safe routes, blocked roads) and interactive legend.
- **Secondary Grid:**
  - Quick Action cards: Launch Satellite Analysis, Run Flood Simulation, View Evacuation Corridors, Generate Situation Report.
  - Active Scenario Metadata & Sensor Pair details.
  - Recent Analysis summary and empty-state guidance when no dataset is loaded.

### Page 2: Satellite Analysis (`#satellite-analysis`)
- **Primary Purpose:** End-to-end multi-spectral and SAR flood detection pipeline.
- **Organized Workflow Steps:**
  - **Step 1: Study Area (AOI) Selection:** Direct integration with Universal AOI controls (Pre-calibrated presets: Chennai, Assam, Wayanad; live city search; coordinate bounding box; GeoJSON polygon).
  - **Step 2: Satellite Sensor Configuration:** Sentinel-1 C-band SAR (VV/VH), Sentinel-2 MSI Optical RGB/NIR, pre-event and post-event dates, cloud cover slider.
  - **Step 3: Data Acquisition & Provenance:** Live Copernicus/GEE vs Local Cache; file upload pair option; transparent dataset status.
  - **Step 4: Deep Learning Execution:** Dual-Stream Siamese Attention U-Net inference execution with progress feedback.
  - **Step 5: Multi-Modal Results Stage:**
    - Dual Viewport Switcher: Interactive GIS Map vs Split-Screen Curtain Comparator (Pre-Optical vs Post-SAR, NDWI, Cross-Modal Attention Saliency).
    - Hydrology Breakdown Bar Chart ($km^2$).
    - SAR Backscatter Histogram ($-16\text{ dB}$ specular water cutoff).
    - Expandable Technical Details: Model architecture comparison (Sen1Floods11 ablation benchmark), mIoU ($89.2\%$), Dice/F1 ($94.1\%$), Precision, Recall.

### Page 3: Flood Simulation (`#flood-simulation`)
- **Primary Purpose:** Dynamic hydrological modeling and cyclone scenario laboratory.
- **Mode Switcher:**
  1. *Live Disaster Analysis:* Real topography and weather feed.
  2. *Offline Demo Simulation:* Calibrated 48-hour flood surge/recession curve.
  3. *Custom Cyclone Scenario:* Full 2D diffusive wave Saint-Venant simulation.
- **Terrain & DEM Attribution:** Explicit indicator of elevation source (Copernicus GLO-30/90, SRTM, or hydro-conditioned procedural elevation).
- **Cyclone Parameter Groups:**
  - *Group A (Wind):* Maximum sustained wind (km/h), central pressure (hPa), vortex radius $R_{\text{max}}$ (km), forward storm speed (km/h), wind direction (°).
  - *Group B (Rainfall):* Peak rainfall intensity (mm/h), duration (hours), distribution profile (front-loaded, bell-shaped, back-loaded, uniform).
  - *Group C (Numerical Solver):* Simulation horizon (hours), timestep (seconds), Horton infiltration ($f_0, f_c, k$), Manning's roughness $n$.
- **Transparent Parameter Provenance:** Input summary table tagging each parameter's origin (`USER_INPUT`, `PRESET`, `BACKEND_DEFAULT`, `REAL_EXTERNAL`, `SYNTHETIC_ASSUMPTION`).
- **Results & Mass Conservation:** Dynamic flooded area ($km^2$), max depth ($m$), mean depth ($m$), exposed buildings, severed roads, and true physical mass balance error residual ($|\Delta V| / V_{\text{rain}} \times 100\%$).
- **Interactive Timestep Scrubber:** Scrubbing $T+0, T+1, T+3, T+6, T+12, T+24, T+48$ synchronously updates the flood depth map, KPI telemetry, and route statuses.
- **Mandatory Disclaimer:** "SIMULATION ONLY — NOT AN OFFICIAL METEOROLOGICAL FORECAST."

### Page 4: Rescue & Infrastructure (`#rescue`)
- **Primary Purpose:** Emergency evacuation planning, road network integrity, and building damage grading.
- **Infrastructure Exposure Metrics:**
  - Critical Submerged structures ($h \ge 0.15\text{ m}$).
  - Moderate 100m Buffer Risk structures.
  - Passable vs Flooded OSM Highway kilometers.
- **Interactive Routing Map:** Real OpenStreetMap highway lines, color-coded active A* corridors, blocked road collision markers.
- **Dispatch Route Table:** Route ID, origin base, target destination, distance (km), estimated transit time (mins), waypoints count, and navigable status.
- **Dynamic Route Recalculation:** Trigger rerouting against current flood depth raster.
- **Operational Safety Notice:** Explicit distinction between computationally generated candidate corridors and field-verified routes.

### Page 5: Disaster Reports (`#reports`)
- **Primary Purpose:** Automated situation report (SitRep) generation and multi-format export center.
- **SitRep Generator:** Dual-engine generator (Gemini 2.5 Flash with fallback to deterministic expert rule engine).
- **Report Sections:**
  - Executive Disaster Summary & Threat Level.
  - Quantitative Impact Telemetry (Flood extent, infrastructure loss).
  - Exposed Facilities & Priority Evacuation Zones.
  - Tactical Recommendations for NDRF / SDRF / District Collectors.
- **Export Options:**
  - Download Formatted Markdown (`.md`).
  - Download Georeferenced GeoJSON (`.geojson`) containing flood polygons, building grades, and route paths.
  - Download Raw JSON (`.json`) simulation audit report.
  - Browser Print View with clean print styles.

### Page 6: Data Sources (`#data-sources`)
- **Primary Purpose:** Data provider status, live connection health, and offline cache administration.
- **Provider Status Cards:**
  - Copernicus Data Space Ecosystem (Sentinel-1/2, DEM).
  - Google Earth Engine (Optical/SAR composite pipelines).
  - OpenStreetMap / Overpass API (Real highways & building polygons).
  - Open-Meteo Weather API (Real live precipitation & wind).
  - NASA GPM IMERG (Satellite precipitation).
  - Local Offline Cache Storage.
- **Connection & Provenance Indicators:** Authenticated, connected, stub/fallback, or offline.
- **Operational Mode Toggle:** Live, Offline, Hybrid.
- **Cache Administration:** Cache inventory inspection, integrity validation check, and selective cache flush.

### Page 7: Settings (`#settings`)
- **Primary Purpose:** User preferences, hardware telemetry, and defense guides.
- **Supported Settings:**
  - Default Study Area Preset.
  - Default Basemap (Satellite, Dark Tactical, OSM).
  - Default Data Acquisition Mode (Live, Offline, Hybrid).
  - Deep Learning Engine Diagnostics (Device, CUDA, inference latency).
  - Project Defense / Viva Guide launcher.

---

## 4. Design System Specifications

| Token | Hex Value | Application |
|---|---|---|
| **Primary Accent** | `#2563EB` (Blue 600) | Primary buttons, active sidebar items, primary focus |
| **Primary Light** | `#EFF6FF` (Blue 50) | Active background highlights, subtle selection fills |
| **Background** | `#F8FAFC` (Slate 50) | Main application body and canvas |
| **Surface** | `#FFFFFF` (White) | Cards, panels, sidebars, modals |
| **Main Text** | `#0F172A` (Slate 900) | Primary headings, titles, high-contrast labels |
| **Secondary Text**| `#64748B` (Slate 500) | Subtitles, helper text, units, table headers |
| **Border** | `#E2E8F0` (Slate 200) | Card borders, dividers, table grid lines |
| **Success** | `#16A34A` (Green 600) | Passable roads, safe structures, live connections |
| **Warning** | `#D97706` (Amber 600) | 100m buffer risk, moderate damage, cached data |
| **Danger** | `#DC2626` (Red 600) | Flooded areas, severed roads, critical damage |
| **Info / Cyan** | `#0891B2` (Cyan 600) | Permanent water, hydrological indicators, radar |
