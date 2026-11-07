# FloodSense-AI: Comprehensive Demonstration & Viva Defense Guide

**Document Reference:** `DEMO_GUIDE.md`  
**Purpose:** Step-by-step viva script, demonstration commands, and examiner question talking points for B.E. Computer Science final-year evaluation.

---

## 1. Quick Start / Viva Launch Command

To launch the complete platform in a single command, run the automated unified batch script:

```bat
run_floodsense.bat
```

This script:
1. Terminates any lingering processes on ports `8000` (FastAPI) and `5173` (Vite).
2. Starts the FastAPI backend daemon with live reload at `http://localhost:8000`.
3. Starts the React / Vite frontend development server at `http://localhost:5173`.
4. Automatically opens `http://localhost:5173` in the default web browser.

---

## 2. Step-by-Step Viva Demonstration Script (10-Minute Walkthrough)

### Minute 1–2: Introduction & Problem Statement
- **Open Dashboard:** Navigate to `http://localhost:5173`.
- **Examiner Pitch:**
  > *"Respected Examiners, during catastrophic flood disasters such as the Chennai monsoons or Assam Brahmaputra floods, optical satellites fail due to heavy cloud cover, and static road maps are rendered useless as floodwaters submerge arterial corridors. FloodSense-AI solves this through an end-to-end multi-modal deep learning and dynamic emergency routing pipeline that combines cloud-penetrating Sentinel-1 SAR, Sentinel-2 Optical, real OpenStreetMap vectors, and 2D mass-conservative hydraulic simulation."*

### Minute 3–4: Multi-Modal Optical vs. SAR Inspection
- **Action:** In the top bar, switch the viewport tab from **Tactical Map View** to **Before / After Split Compare**.
- **Demonstration:**
  - Drag the interactive curtain slider left and right across the Adyar River basin.
  - Show the left pane: Pre-Flood Sentinel-2 True Color Optical imagery showing calm perennial channels and dense urban blocks.
  - Show the right pane: Post-Flood Sentinel-1 SAR backscatter where calm floodwaters appear dark due to specular microwave reflection away from the radar sensor.
  - Explain the NDWI formula and SAR backscatter drop ($<-16\text{ dB}$).

### Minute 5–6: Deep Learning Inference & OpenStreetMap Damage Mapping
- **Action:** Switch back to **Tactical Map View**.
- **Demonstration:**
  - Point out the tri-class semantic segmentation overlay:
    - **Cyan:** Baseline Permanent Water Body (Class 1)
    - **Rose Red:** Active Flood Inundation Zone (Class 2)
    - **Amber:** 100m High-Risk Buffer Zone
  - Toggle the **Layer Visibility** checkboxes in the left sidebar: show how individual layers (`Active Flood`, `100m Risk Buffer`, `Real OSM Roads`, `Building Footprints`) toggle independently.
  - Point out that **5,272 real OSM road segments** and **27,000+ building footprints** are georeferenced via local UTM projections.
  - Highlight the 4-tier road hazard classification (`OPEN`, `CAUTION`, `HIGH_RISK`, `BLOCKED`).

### Minute 7–8: Dynamic 48-Hour Flood Simulation & Live Rescue Routing
- **Action:** Scroll down to the **48-Hour Hydrological Forecaster** in the bottom panel.
- **Demonstration:**
  - Drag the **Rainfall Intensity Slider** from $20\text{ mm/hr}$ up to $135\text{ mm/hr}$ (Cloudburst).
  - Click on the forecast horizon buttons: `T+0h`, `+6h`, `+12h`, `+24h`, `+48h`.
  - Explain the 2D diffusive wave shallow-water solver ($\frac{\partial h}{\partial t} + \nabla \cdot q = R - I - D$).
  - Emphasize the **$0.00\%$ Mass Conservation Error**:
    > *"Unlike naive computer vision dilation which artificially fabricates water pixels, our simulation strictly conserves water mass using physical Manning friction and adaptive flux limiters."*
  - Show the green A\* rescue routes dynamically adapting around severed roads.

### Minute 9–10: Automated NDRF Situation Report & Walkthrough Mode
- **Action 1:** Click the purple **"Viva Guide"** button in the top header.
  - Step through the 7 project review slides to demonstrate thorough architectural mastery.
- **Action 2:** Click the red **"Generate NDRF Action SitRep"** button.
  - Show the structured NDRF Tactical Flash Bulletin.
  - Point out the 4 concrete operational directives: Directive Alpha (Maritime Boat Deployment), Directive Bravo (Evacuation Convoy Routing), Directive Charlie (Traffic Cordons), Directive Delta (Medical Logistics).
  - Click **Download Markdown (.md)** to show instantaneous operational file generation.

---

## 3. High-Scoring Examiner Q&A Talking Points

**Q1: Why did you use a Dual-Stream Siamese U-Net rather than simply concatenating optical and SAR into a 6-channel input?**  
*Answer:* Optical surface reflectance is a normalized dimensionless ratio $\in [0, 1]$, while SAR backscatter is logarithmic microwave power in decibels $\in [-30, +5]\text{ dB}$. Early concatenation forces initial $3 \times 3$ convolutional filters to jointly convolve disparate physical distributions, inducing gradient instability and high speckle sensitivity. Our dual-stream architecture encodes each modality in a dedicated branch and aligns them via Cross-Modal Attention Fusion (CMAF).

**Q2: How does your system guarantee that the flood simulation is physical and not just graphical dilation?**  
*Answer:* We solve the 2D depth-averaged hydrodynamic continuity equation combined with Manning's diffusive wave flow equation. Every time-step accounts for precipitation inflow, Horton soil infiltration, and urban stormwater drain capacity. An adaptive outflow limiter mathematically enforces that water leaving a cell cannot exceed available water volume, maintaining a $0.00\%$ numerical mass-balance error across 48 hours.

**Q3: How do you handle OpenStreetMap rate limits during an emergency?**  
*Answer:* We employ a dual-tier strategy: the system first attempts an Overpass QL query; if rate-limited or offline, it seamlessly falls back to the direct OSM API 0.6 bbox query and our persistent local GeoJSON disk cache (`backend/data/osm_cache/`).

