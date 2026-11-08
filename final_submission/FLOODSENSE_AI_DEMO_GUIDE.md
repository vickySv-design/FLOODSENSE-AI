# FloodSense-AI: Comprehensive Offline Demonstration Guide & Runbook

**Academic Final Year Project Demonstration Manual**  
**Degree:** Bachelor of Engineering (B.E.) in Computer Science and Engineering  
**Project:** FloodSense-AI: Multi-Modal Flood Detection, Flood Dynamics, Infrastructure Exposure Mapping and Emergency Rescue Routing  
**Repository:** `c:\Users\vicky\Desktop\Disaster Response`  
**Target Environment:** Local Windows 11 Pro, NVIDIA RTX 4070 Laptop GPU (8GB VRAM), Python 3.10 CUDA Venv, Node.js v20+  
**Cost:** $0.00 (100% Offline Local Execution, Zero Paid APIs, Zero Cloud Subscriptions)

---

## 1. System Prerequisites & Environment Verification

Before launching the demonstration, ensure the following local environment components are intact:

| Component | Verified Local Path / Specification | Purpose |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Pro 64-bit | Local workstation host |
| **GPU Hardware** | NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM) | CUDA tensor acceleration for UNet |
| **Python Virtual Env** | `c:\Users\vicky\Desktop\Disaster Response\backend\.venv_cuda` | PyTorch 2.5.1+cu121, Rasterio, GeoPandas, FastAPI |
| **Frontend Runtime** | Node.js v20+ / npm in `c:\Users\vicky\Desktop\Disaster Response\frontend` | React 18, Vite 5, Tailwind CSS, Leaflet |
| **Model Weights** | `backend/outputs/sen1floods11_training/best_checkpoint.pth` (8.24M params)<br>`backend/data/checkpoints/dual_stream_attention_unet.pth` (40.6M params) | Model B (Sen1Floods11 Benchmark)<br>Model A (Synthetic Baseline) |
| **Demonstration Cache** | `backend/data/satellite_cache/` | Pre-downloaded Sentinel-1 & 2 GeoTIFF rasters |

---

## 2. Quick Start: Cold Boot to Demonstration in 60 Seconds

### Step 1: Launch FastAPI Backend (Terminal 1)
Open PowerShell in the project root:
```powershell
cd "c:\Users\vicky\Desktop\Disaster Response\backend"
.\.venv_cuda\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Expected Terminal Output:*
```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```
*Health Check:* Open `http://127.0.0.1:8000/docs` in your browser to verify Swagger interactive API documentation.

---

### Step 2: Launch React Frontend (Terminal 2)
Open a second PowerShell window:
```powershell
cd "c:\Users\vicky\Desktop\Disaster Response\frontend"
npm run dev
```
*Expected Terminal Output:*
```text
  VITE v5.4.19  ready in 420 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

---

### Step 3: Open Dashboard
Navigate to `http://localhost:5173` in Google Chrome or Microsoft Edge.  
The application loads instantaneously in dark mode with 8 modular navigation tabs in the left sidebar.

---

## 3. Step-by-Step Demonstration Walkthrough (Tab by Tab)

### Tab 1: Overview Dashboard (`#dashboard`)
*Talking Point:* "This is the primary situational awareness screen summarizing the current disaster state."
1. Point out the **Key Metric Cards** at the top:
   - **Total Mapped Flood Extent:** 21.05 km² (reconciled geodesic area).
   - **High-Exposure Buildings:** 290 residential/commercial structures.
   - **Hazardous Road Segments:** 504 roadway cuts.
   - **Active Pipeline State:** Idle / Ready.
2. Review the interactive overview map centered on the Adyar River basin (Chennai, India).
3. Demonstrate layer toggles: Switch between OpenStreetMap standard basemap, flood inundation mask, and critical infrastructure markers.

---

### Tab 2: Pipeline Execution (`#pipeline`)
*Talking Point:* "Here we can trigger the complete 8-stage automated disaster intelligence pipeline."
1. Click the **Pipeline Execution** tab in the sidebar.
2. Select the preset **"Chennai Adyar Basin (Demonstration Testbed)"**.
3. Click the blue **"Execute Full Pipeline"** button.
4. Observe the real-time **Server-Sent Events (SSE)** execution stepper:
   - **Stage 1 (Acquisition):** Ingests/loads Sentinel-1 SAR and Sentinel-2 optical scenes.
   - **Stage 2 (Preprocessing):** Converts SAR DN to dB backscatter, normalizes reflectance, calculates NDWI.
   - **Stage 3 (Inference):** Executes Dual-Stream Cross-Attention UNet on NVIDIA RTX 4070 GPU.
   - **Stage 4 (Dynamics):** Evaluates pre- vs post-event SAR backscatter differencing.
   - **Stage 5 (Building Exposure):** Buffers and classifies 1,000 OSM building parcels.
   - **Stage 6 (Road Hazard):** Intersects 1,000 road segments against flood mask in NetworkX DiGraph.
   - **Stage 7 (Rescue Routing):** Computes flood-penalized A* evacuation route.
   - **Stage 8 (Finalization):** Generates immutable execution manifest JSON.
5. Inspect the live scrolling **Logs Console** showing millisecond-level telemetry.

---

### Tab 3: Satellite Analysis (`#satellite-analysis`)
*Talking Point:* "This screen demonstrates the multi-modal physics of active radar versus passive optical sensing."
1. Use the **Split-Screen Swipe Comparison Tool**:
   - **Left View:** Sentinel-1 C-band SAR (VV/VH backscatter). Point out how smooth open water specularly reflects radar energy away from the sensor, appearing dark black/deep blue.
   - **Right View:** Sentinel-2 optical true color (RGB) and false-color infrared.
2. Switch optical overlay to **NDWI (Normalized Difference Water Index)**:
   - Observe how the green/NIR ratio delineates surface water, but explain how optical sensors are blinded during storms, whereas SAR operates 24/7 through clouds.

---

### Tab 4: Flood Simulation (`#flood-simulation`)
*Talking Point:* "Civil defense authorities can perform what-if hydrological risk modeling before peak rainfall."
1. Navigate to the **Flood Simulation** tab.
2. Adjust the **Precipitation Accumulation Slider** (e.g., from 150 mm to 300 mm).
3. Adjust the **Soil Infiltration / Drainage Capacity Slider**.
4. Click **"Run Hydrological Simulation"**.
5. Observe the dynamically simulated inundation contour expanding along low-lying river tributaries and drainage canals.

---

### Tab 5: Candidate Rescue Routing (`#rescue`)
*Talking Point:* "This is the operational decision-support module that routes emergency rescue convoys safely around inundated roadways."
1. View the **Origin Depot** (e.g., Tambaram Air Force Rescue Base) and **Destination Relief Shelter** (e.g., Guindy Emergency Evacuation Center).
2. Click **"Calculate Optimal Rescue Route"**.
3. Highlight the route metrics displayed in the telemetry card:
   - **Total Route Distance:** 3,469.2 meters.
   - **Assumed Convoy Speed:** 25 km/h (standard for heavy rescue trucks in urban waterlogged terrain).
   - **Estimated Time of Arrival (ETA):** Exact 8.33 minutes (calculated as `(3.4692 km / 25 km/h) * 60`).
4. Note that the route deliberately avoids red road segments (Hazard > 30% water) and navigates via green (Clear < 10% water) segments.
5. Point out the **Directed DiGraph Traffic Rule Enforcement**: The route strictly obeys OSM `oneway` tags and does not traverse one-way streets against traffic.

---

### Tab 6: Reports & Model Provenance (`#reports`)
*Talking Point:* "Academic rigor requires complete data provenance and mathematical reconciliation."
1. Inspect the **Model Provenance Card**:
   - **Model B (Sen1Floods11 Benchmark):** 8.24M parameters, trained on official Sen1Floods11 dataset for 28 epochs, achieving 95.50% test accuracy and 0.7198 test mIoU.
   - **Model A (Demonstration Baseline):** 40.6M parameters, calibrated on 24 procedural synthetic tiles for UI integration.
2. Review the **Forensic Area Reconciliation Table**:
   - Centroid Geodesic: **21.050 km²** (2,104.97 ha).
   - Row-Integrated Geodesic: **21.049 km²** (2,104.93 ha).
   - Projected Metric UTM 44N: **21.025 km²** (2,102.46 ha).
   - Legacy Planar Approximation: **21.584 km²** (2,158.39 ha) — explained as a legacy planar approximation omitting cosine latitude convergence.

---

### Tab 7: Data Sources (`#data-sources`)
*Talking Point:* "Shows how open-access earth observation data is ingested without any paid subscriptions."
1. Review the listed satellite sources: Copernicus Sentinel-1A/B SAR and Sentinel-2A/B MSI.
2. Inspect the Overpass API query parameters used to pull OSM vector layers offline.

---

### Tab 8: Settings & Diagnostics (`#settings`)
*Talking Point:* "Provides hardware health telemetry and execution tuning."
1. Inspect the **NVIDIA CUDA GPU Status** card: shows RTX 4070 Laptop GPU VRAM allocation (typically ~1.2 GB idle, ~3.8 GB active during batch inference).
2. Point out the **Clear Output Cache** button for reset and reproducibility.

---

## 4. Addressing Common Examiner Questions During Demo

### Q: "Why does the temporal dynamics module show 0 change pixels during the demo?"
**Examiner Answer:**
> "That is by deliberate scientific design for offline demonstration stability. The Chennai demonstration preset packages a self-contained single-pass satellite tile pair where both pre- and post-event SAR rasters point to the validated flood event scene. Rather than fabricating synthetic changes, our pipeline detects ΔVV = 0.0 dB and transparently logs an informational notice. The dynamic temporal engine was fully validated on the multi-temporal Bolivia holdout dataset where dynamic flood expansion was formally verified."

### Q: "Does the building exposure module prove structural collapse?"
**Examiner Answer:**
> "No, and our documentation explicitly highlights this boundary. The module measures two-dimensional raster intersection between the building footprint polygon and satellite flood pixels. It identifies exposure (the structure is surrounded by floodwater). Whether water enters the building depends on plinth height and architectural materials, which satellite sensors cannot observe."

### Q: "Can rescue vehicles drive through roads marked Clear?"
**Examiner Answer:**
> "Clear indicates that satellite radar detects no surface water covering the roadway. However, as noted in our Candidate Routing Disclosures, satellite remote sensing cannot verify physical bridge load-bearing integrity, subterranean erosion, or local obstructions like fallen trees or downed power lines. Field reconnaissance must always accompany candidate routing."

---

## 5. Troubleshooting & Maintenance Runbook

| Symptom | Probable Cause | One-Line Fix Command |
| :--- | :--- | :--- |
| **Backend fails to start: Port 8000 in use** | Stray Python process | `powershell -Command "Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force"` |
| **Frontend fails to start: Port 5173 in use** | Stray Vite process | `powershell -Command "Get-Process -Id (Get-NetTCPConnection -LocalPort 5173).OwningProcess | Stop-Process -Force"` |
| **PyTorch CUDA out of memory** | VRAM fragmented from long runs | Restart backend process; GPU allocations are freed upon process termination |
| **Browser shows blank/white screen** | Frontend unhandled promise | Check browser DevTools console (F12). ErrorBoundary automatically catches errors and allows one-click reload |
| **Verify all tests pass before demo** | Automated test regression | `backend\.venv_cuda\Scripts\pytest backend/tests/ -v` (Must report 61 passed) |

---

*End of Demonstration Guide. Ready for final examination review.*

