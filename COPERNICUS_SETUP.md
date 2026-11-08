# European Space Agency (ESA) Copernicus Data Space Setup Guide
## Free Academic & Research Access | Zero-Cost Deployment

[![Provider](https://img.shields.io/badge/Provider-Copernicus%20Data%20Space-06b6d4.svg)](#)
[![Cost](https://img.shields.io/badge/Cost-100%25%20Free-10b981.svg)](#)
[![Authentication](https://img.shields.io/badge/Auth-OAuth2%20Client%20Credentials-2563eb.svg)](#)

---

## 1. Overview

The **Copernicus Data Space Ecosystem (CDSE)** is the European Commission and European Space Agency's primary platform for accessing open, all-weather Earth Observation satellite imagery. FloodSense-AI utilizes CDSE to retrieve:
- **Sentinel-2 Multi-Spectral Instrument (MSI):** Level-2A surface reflectance (Bands B02, B03, B04, B08).
- **Sentinel-1 C-Band SAR:** Level-1 Ground Range Detected (GRD) in Interferometric Wide (IW) mode (Polarizations VV, VH).
- **Copernicus Digital Elevation Model (DEM):** GLO-30 / GLO-90 open topography rasters.

All satellite data accessed through CDSE is **completely free of charge** under the Copernicus Open Access Policy.

---

## 2. Step-by-Step Account Registration

1. **Visit the CDSE Registration Portal:**
   Navigate to [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/) and click **Register**.
2. **Fill in Free Account Details:**
   Enter your name, academic institution/email, and select "Education & Research" as your usage category. No credit card or payment information is ever requested.
3. **Confirm Email:**
   Click the activation link sent to your registered email address.

---

## 3. Generating API OAuth2 Client Credentials

To query the CDSE catalogue and download raster bands programmatically via FloodSense-AI:
1. Log in to your CDSE account dashboard: [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)
2. Open the user profile dropdown in the top-right corner and select **User Dashboard**.
3. Under the **API Settings** tab, scroll to **OAuth2 Client Credentials**.
4. Click **Create New Client**.
5. Copy your **Client ID** and **Client Secret**.

---

## 4. Configuring Backend Environment Variables

Open (or create) the `.env` file in the root directory:

```env
# European Space Agency (ESA) Copernicus Data Space Ecosystem (CDSE)
COPERNICUS_CLIENT_ID=your_client_id_here
COPERNICUS_CLIENT_SECRET=your_client_secret_here

# Alternative (Username / Password)
# COPERNICUS_USERNAME=your_username
# COPERNICUS_PASSWORD=your_password

# Operating Mode: HYBRID, LIVE, or OFFLINE
DATA_MODE=HYBRID
CACHE_ENABLED=true
```

---

## 5. API Endpoints Used by FloodSense-AI

| Service | Canonical URL | Method | Purpose in FloodSense-AI |
|---|---|:---:|---|
| **Identity Service** | `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token` | `POST` | Exchanges client credentials for OAuth2 Bearer token ($3600\text{s}$ lifetime). |
| **OData Catalogue** | `https://catalogue.dataspace.copernicus.eu/odata/v1/Products` | `GET` | Spatial and temporal search filtering by AOI polygon WKT and date. |
| **OData Zipper API** | `https://catalogue.dataspace.copernicus.eu/odata/v1/Products({id})/$value` | `GET` | Direct product stream and granule band extraction. |
| **STAC Catalogue** | `https://catalogue.dataspace.copernicus.eu/stac` | `GET` | SpatioTemporal Asset Catalog metadata discovery. |

---

## 6. Verifying Live Connection

You can verify your CDSE credentials using the automated test suite or via curl:

```bash
# Direct diagnostic check using FloodSense-AI Python service:
python -c "from app.services.data_providers.copernicus_provider import CopernicusDataSpaceProvider; p = CopernicusDataSpaceProvider(); print(p.check_connection())"
```

Expected output with configured credentials:
```json
{
  "provider": "copernicus",
  "status": "CONNECTED",
  "connected": true,
  "message": "Successfully authenticated with European Space Agency CDSE API.",
  "latency_ms": 320.5,
  "auth_type": "OAuth2 Client Credentials"
}
```

---

## 7. Zero-Cost Offline Fallback Behavior

If credentials are not configured or if the user is operating without internet access:
- FloodSense-AI automatically logs:
  `[INFO] Copernicus CDSE credentials not configured in environment. Running in HYBRID/OFFLINE mode.`
- Public OData searches continue to match valid satellite acquisition dates.
- Actual raster pixels are loaded from the **verified local genuine cache** (`backend/data/satellite_cache` and `backend/data/presets`), generating validated GeoTIFFs via `rasterio`.
- Downstream deep-learning inference and rescue routing continue without interruption.
