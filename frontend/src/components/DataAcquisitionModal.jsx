import React, { useState, useEffect } from "react";
import {
  Satellite,
  Mountain,
  Building2,
  CloudRain,
  Database,
  X,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  RefreshCw,
  Download,
  Search,
  ExternalLink,
  ShieldCheck,
  Radio,
} from "lucide-react";
import {
  fetchDataProvidersStatus,
  searchSatelliteCatalog,
  startDataDownloadJob,
  fetchDEMData,
  fetchOSMData,
  fetchRainfallData,
  fetchGPMData,
  fetchDataJobStatus,
  fetchDataCache,
  validateDataCache,
} from "../services/api.js";

export default function DataAcquisitionModal({
  isOpen,
  onClose,
  currentAOI,
  onDataAcquired,
}) {
  const [activeTab, setActiveTab] = useState("satellite"); // 'satellite' | 'terrain' | 'infrastructure' | 'rainfall' | 'cache'
  const [systemMode, setSystemMode] = useState("hybrid"); // 'live' | 'offline' | 'hybrid'
  const [providersStatus, setProvidersStatus] = useState(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState(false);

  // Satellite Tab State
  const [satellitePlatform, setSatellitePlatform] = useState("SENTINEL-2");
  const [dateRange, setDateRange] = useState({
    start: "2025-11-01",
    end: "2025-12-10",
  });
  const [cloudCoverMax, setCloudCoverMax] = useState(20);
  const [searchResults, setSearchResults] = useState([]);
  const [selectedScenes, setSelectedScenes] = useState([]);
  const [isSearchingSatellite, setIsSearchingSatellite] = useState(false);

  // Terrain Tab State
  const [demType, setDemType] = useState("GLO-30");
  const [demInfo, setDemInfo] = useState(null);
  const [isFetchingDEM, setIsFetchingDEM] = useState(false);

  // Infrastructure Tab State
  const [osmInfo, setOsmInfo] = useState(null);
  const [isFetchingOSM, setIsFetchingOSM] = useState(false);

  // Rainfall Tab State
  const [rainfallSource, setRainfallSource] = useState("open_meteo"); // 'open_meteo' | 'nasa_gpm'
  const [rainfallData, setRainfallData] = useState(null);
  const [isFetchingRainfall, setIsFetchingRainfall] = useState(false);

  // Cache Tab State
  const [cacheAudit, setCacheAudit] = useState(null);
  const [isValidatingCache, setIsValidatingCache] = useState(false);

  // Active Job Tracking
  const [activeJobId, setActiveJobId] = useState(null);
  const [jobProgress, setJobProgress] = useState(null);

  const bbox = currentAOI?.bbox || [12.980, 80.190, 13.060, 80.290];
  const aoiName = currentAOI?.name || "Selected Study Area";

  useEffect(() => {
    if (!isOpen) return;
    loadStatuses();
    loadCacheAudit();
  }, [isOpen]);

  // Poll active acquisition job
  useEffect(() => {
    if (!activeJobId) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetchDataJobStatus(activeJobId);
        setJobProgress(res);
        if (res.status === "COMPLETED" || res.status === "FAILED") {
          setActiveJobId(null);
          loadCacheAudit();
          if (res.status === "COMPLETED" && onDataAcquired) {
            onDataAcquired(res.results);
          }
        }
      } catch (err) {
        setActiveJobId(null);
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [activeJobId]);

  const loadStatuses = async () => {
    setIsLoadingStatus(true);
    try {
      const res = await fetchDataProvidersStatus();
      setProvidersStatus(res);
      if (res.mode) setSystemMode(res.mode);
    } catch (err) {
      console.warn("Failed to load provider statuses", err);
    } finally {
      setIsLoadingStatus(false);
    }
  };

  const loadCacheAudit = async () => {
    try {
      const res = await fetchDataCache();
      setCacheAudit(res);
    } catch (err) {
      console.warn("Failed to load cache audit", err);
    }
  };

  const handleSearchSatellite = async (e) => {
    e?.preventDefault();
    setIsSearchingSatellite(true);
    try {
      const res = await searchSatelliteCatalog({
        provider: "copernicus",
        bounds: bbox,
        start_date: dateRange.start,
        end_date: dateRange.end,
        platform: satellitePlatform,
        max_cloud_cover_pct: Number(cloudCoverMax),
      });
      setSearchResults(res.scenes || []);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSearchingSatellite(false);
    }
  };

  const handleStartDownload = async () => {
    try {
      const res = await startDataDownloadJob({
        aoi_id: currentAOI?.id || "custom_aoi",
        bbox: bbox,
        product_ids: selectedScenes,
        include_dem: true,
        include_osm: true,
        include_rainfall: true,
      });
      setActiveJobId(res.job_id);
    } catch (err) {
      console.error(err);
    }
  };

  const handleFetchDEM = async () => {
    setIsFetchingDEM(true);
    try {
      const res = await fetchDEMData({ bbox, dem_type: demType });
      setDemInfo(res);
      loadCacheAudit();
    } catch (err) {
      console.error(err);
    } finally {
      setIsFetchingDEM(false);
    }
  };

  const handleFetchOSM = async () => {
    setIsFetchingOSM(true);
    try {
      const res = await fetchOSMData({
        aoi_id: currentAOI?.id || "custom_aoi",
        bbox,
        force_refresh: true,
      });
      setOsmInfo(res);
      loadCacheAudit();
    } catch (err) {
      console.error(err);
    } finally {
      setIsFetchingOSM(false);
    }
  };

  const handleFetchRainfall = async () => {
    setIsFetchingRainfall(true);
    try {
      if (rainfallSource === "open_meteo") {
        const res = await fetchRainfallData({ bbox, mode: "forecast", forecast_days: 3 });
        setRainfallData(res);
      } else {
        const res = await fetchGPMData({ bbox, product: "early", hours: 24 });
        setRainfallData(res);
      }
      loadCacheAudit();
    } catch (err) {
      console.error(err);
    } finally {
      setIsFetchingRainfall(false);
    }
  };

  const handleValidateCache = async () => {
    setIsValidatingCache(true);
    try {
      const res = await validateDataCache();
      setCacheAudit(res.cache_audit);
    } catch (err) {
      console.error(err);
    } finally {
      setIsValidatingCache(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-md flex items-center justify-center p-4 overflow-y-auto">
      <div className="w-full max-w-4xl bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh]">
        {/* Header */}
        <div className="px-5 py-3.5 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-500/15 border border-emerald-500/40 text-emerald-400">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-mono font-bold text-slate-100 uppercase tracking-wider">
                  Live Data API Integration &amp; Offline Caching
                </h2>
                {/* Operating Mode Pill */}
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded-full border font-bold uppercase ${
                    systemMode === "live"
                      ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40"
                      : systemMode === "offline"
                      ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                      : "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                  }`}
                >
                  Mode: {systemMode.toUpperCase()}
                </span>
              </div>
              <p className="text-xs text-slate-400">
                AOI: <span className="text-slate-200 font-semibold">{aoiName}</span> | BBox: [
                {bbox.map((b) => b.toFixed(2)).join(", ")}]
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Mode Selector & Provider Status Bar */}
        <div className="px-5 py-2.5 bg-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="text-slate-400 text-[11px]">Data Mode:</span>
            {["hybrid", "live", "offline"].map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setSystemMode(m)}
                className={`px-2.5 py-1 rounded text-[11px] font-semibold transition ${
                  systemMode === m
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                    : "bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200"
                }`}
              >
                {m.toUpperCase()}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3 text-[11px]">
            <span className="text-slate-500 flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              Free Academic APIs
            </span>
            <button
              type="button"
              onClick={loadStatuses}
              disabled={isLoadingStatus}
              className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 transition"
            >
              <RefreshCw className={`w-3 h-3 ${isLoadingStatus ? "animate-spin" : ""}`} />
              Refresh Status
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-slate-800 bg-slate-950/60 px-5 text-xs font-mono">
          {[
            { id: "satellite", label: "Copernicus Satellite", icon: Satellite },
            { id: "terrain", label: "Copernicus DEM", icon: Mountain },
            { id: "infrastructure", label: "OpenStreetMap", icon: Building2 },
            { id: "rainfall", label: "Rainfall (Open-Meteo/GPM)", icon: CloudRain },
            { id: "cache", label: "Local Cache & Provenance", icon: Database },
          ].map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 py-3 px-3.5 border-b-2 font-bold transition ${
                  active
                    ? "border-cyan-400 text-cyan-300 bg-slate-900/50"
                    : "border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/25"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Modal Body */}
        <div className="flex-1 p-5 overflow-y-auto space-y-4 text-xs font-mono">
          {/* TAB 1: SATELLITE (COPERNICUS CDSE) */}
          {activeTab === "satellite" && (
            <div className="space-y-4">
              <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Satellite className="w-4 h-4 text-cyan-400" />
                    <span className="font-bold text-slate-200">
                      European Space Agency (ESA) Copernicus Data Space
                    </span>
                  </div>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      providersStatus?.providers?.copernicus?.status === "CONNECTED"
                        ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                        : "bg-cyan-500/10 text-cyan-300 border border-cyan-500/30"
                    }`}
                  >
                    {providersStatus?.providers?.copernicus?.status || "PUBLIC CATALOGUE QUERY"}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Direct programmatic discovery for Sentinel-1 C-SAR (GRD dual-pol VV/VH) and Sentinel-2 MSI (Level-2A
                  surface reflectance) matching the exact configured AOI bounds.
                </p>
              </div>

              {/* Search Form */}
              <form onSubmit={handleSearchSatellite} className="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div>
                  <label className="text-[10px] text-slate-400 uppercase">Platform</label>
                  <select
                    value={satellitePlatform}
                    onChange={(e) => setSatellitePlatform(e.target.value)}
                    className="w-full mt-1 bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200"
                  >
                    <option value="SENTINEL-2">Sentinel-2 (Optical MSI L2A)</option>
                    <option value="SENTINEL-1">Sentinel-1 (Radar C-SAR GRD)</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] text-slate-400 uppercase">Start Date</label>
                  <input
                    type="date"
                    value={dateRange.start}
                    onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
                    className="w-full mt-1 bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-slate-200"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-slate-400 uppercase">End Date</label>
                  <input
                    type="date"
                    value={dateRange.end}
                    onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
                    className="w-full mt-1 bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-slate-200"
                  />
                </div>
                <div className="flex items-end">
                  <button
                    type="submit"
                    disabled={isSearchingSatellite}
                    className="w-full py-1.5 px-3 rounded bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold transition flex items-center justify-center gap-1.5 disabled:opacity-50"
                  >
                    {isSearchingSatellite ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
                    Search Scenes
                  </button>
                </div>
              </form>

              {/* Results List */}
              {searchResults.length > 0 && (
                <div className="space-y-2">
                  <div className="text-[11px] text-slate-400 flex items-center justify-between">
                    <span>Discovered Scenes ({searchResults.length}):</span>
                    <button
                      type="button"
                      onClick={() => setSelectedScenes(searchResults.map((s) => s.scene_id))}
                      className="text-cyan-400 hover:underline"
                    >
                      Select All
                    </button>
                  </div>
                  <div className="max-h-48 overflow-y-auto space-y-1.5">
                    {searchResults.map((s) => (
                      <div
                        key={s.scene_id}
                        className="p-2.5 rounded bg-slate-950 border border-slate-800 flex items-center justify-between"
                      >
                        <div>
                          <div className="text-slate-200 font-bold">{s.scene_id}</div>
                          <div className="text-[10px] text-slate-500">
                            Date: {s.acquisition_date?.slice(0, 10)} | Platform: {s.platform} | Level: {s.product_level}
                            {s.cloud_cover_pct !== null && ` | Cloud: ${s.cloud_cover_pct}%`}
                          </div>
                        </div>
                        <input
                          type="checkbox"
                          checked={selectedScenes.includes(s.scene_id)}
                          onChange={(e) => {
                            if (e.target.checked) setSelectedScenes([...selectedScenes, s.scene_id]);
                            else setSelectedScenes(selectedScenes.filter((id) => id !== s.scene_id));
                          }}
                          className="w-4 h-4 rounded text-cyan-500"
                        />
                      </div>
                    ))}
                  </div>

                  <button
                    type="button"
                    onClick={handleStartDownload}
                    disabled={selectedScenes.length === 0 || Boolean(activeJobId)}
                    className="w-full py-2 rounded bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold transition flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    <Download className="w-4 h-4" />
                    Download Selected ({selectedScenes.length}) &amp; Cache Locally
                  </button>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: TERRAIN (COPERNICUS DEM) */}
          {activeTab === "terrain" && (
            <div className="space-y-4">
              <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Mountain className="w-4 h-4 text-emerald-400" />
                    <span className="font-bold text-slate-200">Copernicus Digital Elevation Model (COP-DEM)</span>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                    Open Access
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Retrieves hydro-conditioned 30m (GLO-30) or 90m (GLO-90) digital elevation rasters. Directly feeds
                  into the 2D shallow-water flood simulation solver to govern gravitational flow directions.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex-1">
                  <label className="text-[10px] text-slate-400 uppercase">Product Resolution</label>
                  <select
                    value={demType}
                    onChange={(e) => setDemType(e.target.value)}
                    className="w-full mt-1 bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200"
                  >
                    <option value="GLO-30">Copernicus GLO-30 (30m High Resolution)</option>
                    <option value="GLO-90">Copernicus GLO-90 (90m Regional Resolution)</option>
                    <option value="SRTM-30">NASA SRTM 30m Global Elevation</option>
                  </select>
                </div>
                <div className="flex items-end">
                  <button
                    type="button"
                    onClick={handleFetchDEM}
                    disabled={isFetchingDEM}
                    className="py-1.5 px-4 rounded bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold transition flex items-center gap-1.5 disabled:opacity-50"
                  >
                    {isFetchingDEM ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
                    Acquire DEM Raster
                  </button>
                </div>
              </div>

              {demInfo && (
                <div className="p-3 rounded bg-slate-950 border border-slate-800 space-y-1.5">
                  <div className="text-emerald-400 font-bold">DEM Acquired: {demInfo.dem_id}</div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px] text-slate-300">
                    <div>Min Elev: <strong>{demInfo.min_elevation_m?.toFixed(1)} m</strong></div>
                    <div>Max Elev: <strong>{demInfo.max_elevation_m?.toFixed(1)} m</strong></div>
                    <div>Resolution: <strong>{demInfo.resolution_m} m</strong></div>
                    <div>CRS: <strong>{demInfo.crs}</strong></div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: INFRASTRUCTURE (OPENSTREETMAP) */}
          {activeTab === "infrastructure" && (
            <div className="space-y-4">
              <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Building2 className="w-4 h-4 text-purple-400" />
                    <span className="font-bold text-slate-200">OpenStreetMap Live Overpass Client</span>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/10 text-purple-400 border border-purple-500/30">
                    ODbL Attribution
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Downloads real road network polylines (with bridge, tunnel, and one-way tags) and building footprints
                  for topological risk assessment and dynamic A* evacuation routing.
                </p>
              </div>

              <div className="flex items-center justify-between">
                <div className="text-[11px] text-slate-400">
                  Target Sector: <strong className="text-slate-200">{aoiName}</strong>
                </div>
                <button
                  type="button"
                  onClick={handleFetchOSM}
                  disabled={isFetchingOSM}
                  className="py-1.5 px-4 rounded bg-purple-500 hover:bg-purple-400 text-slate-950 font-bold transition flex items-center gap-1.5 disabled:opacity-50"
                >
                  {isFetchingOSM ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                  Fetch &amp; Re-Cache OSM Geometries
                </button>
              </div>

              {osmInfo && (
                <div className="p-3 rounded bg-slate-950 border border-slate-800 space-y-2">
                  <div className="text-purple-400 font-bold">OSM Vectors Ready</div>
                  <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-300">
                    <div className="p-2 rounded bg-slate-900 border border-slate-800">
                      Road Segments: <strong className="text-slate-100">{osmInfo.roads_count}</strong>
                    </div>
                    <div className="p-2 rounded bg-slate-900 border border-slate-800">
                      Building Footprints: <strong className="text-slate-100">{osmInfo.buildings_count}</strong>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 4: RAINFALL (OPEN-METEO / NASA GPM) */}
          {activeTab === "rainfall" && (
            <div className="space-y-4">
              <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CloudRain className="w-4 h-4 text-cyan-400" />
                    <span className="font-bold text-slate-200">Precipitation Telemetry &amp; Forecaster</span>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                    Open-Meteo &amp; NASA GPM
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Explicitly differentiates between atmospheric model forecasts (Open-Meteo) and satellite microwave
                  telemetry observations (NASA GPM IMERG).
                </p>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex-1">
                  <label className="text-[10px] text-slate-400 uppercase">Rainfall Telemetry Source</label>
                  <select
                    value={rainfallSource}
                    onChange={(e) => setRainfallSource(e.target.value)}
                    className="w-full mt-1 bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200"
                  >
                    <option value="open_meteo">Open-Meteo (Hourly Forecast &amp; Historical Model)</option>
                    <option value="nasa_gpm">NASA GPM IMERG (Satellite Microwave Estimates)</option>
                  </select>
                </div>
                <div className="flex items-end">
                  <button
                    type="button"
                    onClick={handleFetchRainfall}
                    disabled={isFetchingRainfall}
                    className="py-1.5 px-4 rounded bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold transition flex items-center gap-1.5 disabled:opacity-50"
                  >
                    {isFetchingRainfall ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                    Fetch Rainfall Series
                  </button>
                </div>
              </div>

              {rainfallData && (
                <div className="p-3 rounded bg-slate-950 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-cyan-400 font-bold">
                    <span>Source: {rainfallData.data_source}</span>
                    <span>Total Accumulation: {rainfallData.total_precipitation_mm} mm</span>
                  </div>
                  <div className="text-[10px] text-slate-400">
                    Peak Intensity: {rainfallData.peak_intensity_mm_hr} mm/h | Center: Lat{" "}
                    {rainfallData.center_coordinates?.latitude}°, Lon {rainfallData.center_coordinates?.longitude}°
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 5: CACHE & PROVENANCE */}
          {activeTab === "cache" && (
            <div className="space-y-4">
              <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Database className="w-4 h-4 text-amber-400" />
                    <span className="font-bold text-slate-200">Local Persistent Cache &amp; Provenance</span>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                    100% Offline Capable
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Audits all stored satellite scenes, hydro-conditioned elevation grids, vector geometries, and weather
                  series. Guarantees complete offline execution during network disconnects.
                </p>
              </div>

              {cacheAudit && (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {Object.entries(cacheAudit.layers || {}).map(([key, val]) => (
                      <div key={key} className="p-2.5 rounded bg-slate-950 border border-slate-800">
                        <div className="text-[10px] text-slate-500 uppercase">{key.replace("_", " ")}</div>
                        <div className="text-sm font-bold text-slate-200">{val.count} files</div>
                        <div className="text-[10px] text-slate-400">{val.size_mb} MB</div>
                      </div>
                    ))}
                  </div>

                  <div className="flex items-center justify-between p-3 rounded bg-slate-950 border border-slate-800">
                    <span className="text-slate-300">
                      Total Storage Footprint: <strong>{cacheAudit.total_cache_size_mb} MB</strong>
                    </span>
                    <button
                      type="button"
                      onClick={handleValidateCache}
                      disabled={isValidatingCache}
                      className="px-3 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs font-bold transition flex items-center gap-1.5"
                    >
                      {isValidatingCache ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ShieldCheck className="w-3.5 h-3.5" />}
                      Validate Cache Integrity
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Job Progress Indicator if active */}
          {jobProgress && (
            <div className="p-3 rounded-lg bg-slate-950 border border-cyan-500/40 space-y-2">
              <div className="flex items-center justify-between text-xs text-cyan-300 font-bold">
                <span>{jobProgress.current_step}</span>
                <span>{jobProgress.progress_pct}%</span>
              </div>
              <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                <div
                  className="bg-cyan-500 h-full transition-all duration-300"
                  style={{ width: `${jobProgress.progress_pct}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3.5 bg-slate-950 border-t border-slate-800 flex items-center justify-between">
          <span className="text-[11px] text-slate-500">
            Complies with European Space Agency, NASA Earthdata &amp; OpenStreetMap ODbL Licences
          </span>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-200 text-xs font-mono font-semibold border border-slate-800 transition"
          >
            Close Panel
          </button>
        </div>
      </div>
    </div>
  );
}
