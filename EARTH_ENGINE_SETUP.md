# Google Earth Engine (GEE) Setup Guide
## Free Academic & Non-Commercial Planetary Computing | Zero-Cost Deployment

[![Provider](https://img.shields.io/badge/Provider-Google%20Earth%20Engine-4285f4.svg)](#)
[![Cost](https://img.shields.io/badge/Cost-100%25%20Free%20Academic-10b981.svg)](#)
[![Library](https://img.shields.io/badge/Python-earthengine--api%201.7.48-06b6d4.svg)](#)

---

## 1. Overview

**Google Earth Engine (GEE)** provides multi-petabyte satellite data catalogs and cloud processing for planetary-scale geospatial analysis. FloodSense-AI interfaces with Earth Engine through the official `earthengine-api` Python library for:
- **Optical Collection:** `COPERNICUS/S2_SR_HARMONIZED` (Level-2A Bottom-of-Atmosphere Reflectance with Scene Classification Cloud Masking).
- **SAR Collection:** `COPERNICUS/S1_GRD` (C-Band Synthetic Aperture Radar in Interferometric Wide mode with VV and VH polarizations).

Academic and research use of Google Earth Engine is **completely free of charge**.

---

## 2. Step-by-Step Free Account Registration

1. **Visit the Earth Engine Sign-Up Portal:**
   Navigate to [https://earthengine.google.com/signup/](https://earthengine.google.com/signup/).
2. **Select Non-Commercial / Academic Project:**
   Choose **"Academic / Research / Student Project"**. Enter your university or college affiliation.
3. **Link to a Google Cloud Project:**
   Google Cloud will automatically prompt you to create or select a free Google Cloud project (e.g. `ee-disaster-response-ai`). No billing account or credit card is required for eligible academic access.

---

## 3. CLI Authentication Flow (Recommended)

Once your Earth Engine account is approved, authenticate directly from your local terminal:

```bash
# In your terminal or PowerShell:
earthengine authenticate
```

1. This opens a web browser window prompting you to sign in with your Google account.
2. Grant read access to Earth Engine.
3. An authorization token will be saved automatically to your local user directory:
   `C:\Users\<user>\.config\earthengine\credentials`

FloodSense-AI will automatically detect these credentials on startup via `ee.Initialize()`.

---

## 4. Alternative: Google Service Account Credentials

For automated headless deployments:
1. Navigate to the Google Cloud Console: `https://console.cloud.google.com/iam-admin/serviceaccounts`
2. Create a service account (e.g. `floodsense-bot@your-project.iam.gserviceaccount.com`).
3. Generate a JSON private key and save it securely on your laptop.
4. Set the following in your `.env` file:

```env
GEE_PROJECT_ID=your-cloud-project-id
GEE_SERVICE_ACCOUNT=floodsense-bot@your-project.iam.gserviceaccount.com
GEE_PRIVATE_KEY_PATH=C:/Users/vicky/keys/gee-key.json
```

---

## 5. Earth Engine Collections & Processing Functions

### 5.1 Sentinel-2 Level-2A (`COPERNICUS/S2_SR_HARMONIZED`)
- **Bands Queried:** `B2` (Blue), `B3` (Green), `B4` (Red), `B8` (Near-Infrared), `SCL` (Scene Classification Layer).
- **Cloud & Shadow Masking Function:**
  ```python
  def mask_s2_clouds(image):
      scl = image.select('SCL')
      # 3: Cloud Shadow, 8: Cloud Medium Prob, 9: Cloud High Prob, 10: Thin Cirrus
      mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
      return image.updateMask(mask)
  ```

### 5.2 Sentinel-1 SAR (`COPERNICUS/S1_GRD`)
- **Instrument Mode:** `IW` (Interferometric Wide).
- **Polarizations:** `VV` (Co-polarization) and `VH` (Cross-polarization).
- **Units:** Radiometric $\sigma^0$ calibrated backscatter in decibels (dB).

### 5.3 Direct GeoTIFF Export via `getDownloadURL`
For interactive study areas ($\le 500\text{ km}^2$), FloodSense-AI avoids slow cloud export tasks by fetching direct GeoTIFF streams:
```python
download_url = clipped_image.getDownloadURL({
    'bands': ['B4', 'B3', 'B2', 'B8'],
    'region': aoi_geometry,
    'scale': 10,
    'format': 'GEO_TIFF'
})
```
The resulting GeoTIFF is downloaded directly to `backend/data/satellite_cache` and validated with `RasterValidator`.

---

## 6. Verifying Connection in FloodSense-AI

Run the diagnostic ping test:
```bash
python -c "from app.services.data_providers.gee_provider import GoogleEarthEngineProvider; p = GoogleEarthEngineProvider(); print(p.check_connection())"
```

Expected output with configured project:
```json
{
  "provider": "gee",
  "status": "CONNECTED",
  "connected": true,
  "message": "Successfully authenticated with Google Earth Engine API.",
  "project_id": "ee-disaster-response-ai",
  "latency_ms": 485.2
}
```

---

## 7. Zero-Cost Offline Fallback Behavior

If `earthengine-api` is unauthenticated or running without an internet connection:
- FloodSense-AI logs `[INFO] GEE credentials not present; running in hybrid/offline mode.`
- Instead of crashing, the system loads the matching verified genuine satellite GeoTIFF from `backend/data/satellite_cache` or `backend/data/presets`.
- The provenance manifest explicitly tags the run as `GENUINE_CACHED_ACQUISITION` or `PRESET_CALIBRATED_BENCHMARK`, maintaining strict scientific integrity.
