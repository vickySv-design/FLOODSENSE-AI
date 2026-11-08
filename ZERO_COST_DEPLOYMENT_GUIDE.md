# FloodSense-AI: Zero-Cost Deployment & Hardware Acceleration Guide
## Local Windows & Linux Execution | NVIDIA RTX GPU Acceleration | Academic Zero-Cost Policy

[![Zero Cost](https://img.shields.io/badge/Policy-100%25%20Zero--Cost-10b981.svg)](#)
[![GPU](https://img.shields.io/badge/Hardware-NVIDIA%20GeForce%20RTX%204070-76b900.svg)](#)
[![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20PyTorch%20%7C%20FastAPI%20%7C%20React-2563eb.svg)](#)

---

## 1. Zero-Cost Academic Architecture Policy

FloodSense-AI was engineered to strictly satisfy academic and collegiate constraints:
- **Zero Paid APIs:** Uses only free, open-access Copernicus CDSE and Google Earth Engine non-commercial tiers.
- **Zero Paid AI Services:** Employs locally executed PyTorch models and offline deterministic NDRF SitRep templates.
- **Zero Paid Cloud Hosting:** Runs 100% locally on a Windows laptop or Linux desktop.
- **Zero Mandatory Subscriptions:** Operates completely offline using cached regional demonstration datasets.

---

## 2. Hardware Acceleration: NVIDIA RTX 4070 Laptop GPU

The machine hosting FloodSense-AI is equipped with an **NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM)**. FloodSense-AI dynamically auto-detects CUDA hardware:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU Engine (PyTorch Optimized)"
```

### Enabling CUDA Acceleration in PyTorch
If running on a CPU-only PyTorch build and desiring GPU acceleration:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Once installed, neural network forward passes drop from $\approx 300\text{ ms}$ (CPU) to **$< 25\text{ ms}$ (RTX GPU)**.

---

## 3. System Prerequisites

1. **Operating System:** Windows 10 / Windows 11 (64-bit) or Ubuntu Linux 20.04+.
2. **Python:** Version 3.10 or 3.11 with `pip`.
3. **Node.js:** Version 18.x or 20.x with `npm`.
4. **RAM:** 8 GB minimum (16 GB recommended).
5. **Disk Storage:** At least 2 GB free disk space for local satellite and OSM caches.

---

## 4. Single-Click Automated Launch (Windows)

Double-click the root batch file:
```cmd
launch.bat
```
This script automatically:
1. Validates Python and Node.js environments.
2. Installs required Python backend packages (`requirements.txt`).
3. Launches the FastAPI backend server on `http://localhost:8000`.
4. Installs frontend packages and launches Vite on `http://localhost:5173`.
5. Automatically opens `http://localhost:5173` in your default browser.

---

## 5. Manual CLI Execution Guide

### Step 1: Backend Setup
```powershell
# Open terminal in project root
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Frontend Setup
```powershell
# In a second terminal window
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` in Google Chrome or Microsoft Edge.

---

## 6. Operating Modes & Configuration

Configured via `.env` in the root workspace:

```env
# Mode 1: HYBRID (Default recommended for college viva)
# Tries live Copernicus / GEE queries first; falls back seamlessly to verified local cache.
DATA_MODE=HYBRID
CACHE_ENABLED=true

# Mode 2: OFFLINE
# Operates completely disconnected from the internet using pre-cached regional presets.
DATA_MODE=OFFLINE

# Mode 3: LIVE
# Strictly queries live satellite APIs; requires configured free CDSE / GEE credentials.
DATA_MODE=LIVE
```

---

## 7. Troubleshooting & FAQ

- **Q: Why does the status show `AUTHENTICATION_REQUIRED`?**  
  *A:* You have not configured `COPERNICUS_CLIENT_ID` or GEE credentials in your `.env`. This is completely normal for offline/hybrid academic demonstrations. The system operates 100% smoothly using verified local genuine GeoTIFFs.
- **Q: Can I run FloodSense-AI without an internet connection during my college viva?**  
  *A:* **Yes.** The platform includes fully verified demonstration datasets for Chennai, Assam, and Wayanad. All rasters, OSM roads, building footprints, and DEM grids are locally cached.
- **Q: Are raster files checked for corruption?**  
  *A:* Yes. The `RasterValidator` service automatically inspects every GeoTIFF with `rasterio`, verifying that dimensions, CRS, affine transforms, and non-zero pixels are physically present.
