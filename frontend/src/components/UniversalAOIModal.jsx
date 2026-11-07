import React, { useState, useEffect } from "react";
import {
  Search,
  MapPin,
  Upload,
  Globe,
  Sliders,
  AlertTriangle,
  CheckCircle2,
  X,
  Loader2,
  Compass,
  Layers,
  ArrowRight,
  Bookmark,
} from "lucide-react";
import { searchAOILocations, validateAOI } from "../services/api.js";

export default function UniversalAOIModal({
  isOpen,
  onClose,
  currentAOI,
  presets,
  onApplyAOI,
}) {
  const [activeTab, setActiveTab] = useState("search"); // 'search' | 'coordinates' | 'upload' | 'presets'
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState(null);

  // Manual Coordinates State
  const [manualName, setManualName] = useState(currentAOI?.name || "Custom Sector");
  const [minLat, setMinLat] = useState(currentAOI?.bbox?.[0] || 12.980);
  const [minLon, setMinLon] = useState(currentAOI?.bbox?.[1] || 80.190);
  const [maxLat, setMaxLat] = useState(currentAOI?.bbox?.[2] || 13.060);
  const [maxLon, setMaxLon] = useState(currentAOI?.bbox?.[3] || 80.290);
  const [terrainType, setTerrainType] = useState(currentAOI?.terrain_type || "auto");

  // Selected AOI Metadata for Preview
  const [previewAOI, setPreviewAOI] = useState(null);
  const [isValidating, setIsValidating] = useState(false);
  const [validationError, setValidationError] = useState(null);

  // Auto-validate whenever coordinates change in manual tab
  useEffect(() => {
    if (!isOpen) return;
    const timer = setTimeout(async () => {
      if (minLat < maxLat && minLon < maxLon) {
        setIsValidating(true);
        setValidationError(null);
        try {
          const res = await validateAOI({
            name: manualName,
            bbox: [Number(minLat), Number(minLon), Number(maxLat), Number(maxLon)],
            terrain_type: terrainType,
          });
          setPreviewAOI(res);
        } catch (err) {
          setValidationError(err.response?.data?.detail || "Validation failed for selected coordinates.");
        } finally {
          setIsValidating(false);
        }
      } else {
        setValidationError("Max latitude and longitude must be strictly greater than Min.");
      }
    }, 250);

    return () => clearTimeout(timer);
  }, [minLat, minLon, maxLat, maxLon, manualName, terrainType, isOpen]);

  // Execute Search via OpenStreetMap Geocoding
  const handleSearch = async (e) => {
    e?.preventDefault();
    if (!searchQuery.trim() || searchQuery.trim().length < 2) return;
    setIsSearching(true);
    setSearchError(null);
    try {
      const results = await searchAOILocations(searchQuery.trim());
      setSearchResults(results);
      if (results.length === 0) {
        setSearchError("No geographical locations matched your search query. Try a city or district name.");
      }
    } catch (err) {
      setSearchError("Location search unavailable. Please enter coordinates manually.");
    } finally {
      setIsSearching(false);
    }
  };

  const handleSelectSearchResult = (loc) => {
    setManualName(loc.name || loc.display_name.split(",")[0]);
    setMinLat(loc.bbox[0]);
    setMinLon(loc.bbox[1]);
    setMaxLat(loc.bbox[2]);
    setMaxLon(loc.bbox[3]);
    setPreviewAOI({
      name: loc.name || loc.display_name.split(",")[0],
      bbox: loc.bbox,
      area_km2: loc.area_km2,
      utm_crs: "Auto-Projected",
      terrain_type: "auto",
    });
    setActiveTab("coordinates");
  };

  const handleSelectPreset = (preset) => {
    setManualName(preset.name);
    setMinLat(preset.bbox[0]);
    setMinLon(preset.bbox[1]);
    setMaxLat(preset.bbox[2]);
    setMaxLon(preset.bbox[3]);
    setTerrainType(preset.terrain_type || "auto");
    setPreviewAOI({
      name: preset.name,
      bbox: preset.bbox,
      preset_id: preset.id,
      area_km2: 95.0,
      utm_crs: "EPSG:32644",
      terrain_type: preset.terrain_type || "auto",
    });
    setActiveTab("coordinates");
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const parsed = JSON.parse(event.target.result);
        let coords = [];
        if (parsed.type === "FeatureCollection" && parsed.features?.[0]) {
          coords = parsed.features[0].geometry.coordinates;
        } else if (parsed.type === "Feature") {
          coords = parsed.geometry.coordinates;
        } else if (parsed.coordinates) {
          coords = parsed.coordinates;
        }

        // Flatten coordinates to find bounding box
        const flat = coords.flat(Infinity);
        const lons = [];
        const lats = [];
        for (let i = 0; i < flat.length; i += 2) {
          if (flat[i] !== undefined && flat[i+1] !== undefined) {
            lons.push(flat[i]);
            lats.push(flat[i+1]);
          }
        }

        if (lons.length >= 3 && lats.length >= 3) {
          const bMinLat = Math.min(...lats);
          const bMaxLat = Math.max(...lats);
          const bMinLon = Math.min(...lons);
          const bMaxLon = Math.max(...lons);

          setManualName(file.name.replace(/\.[^/.]+$/, ""));
          setMinLat(round5(bMinLat));
          setMinLon(round5(bMinLon));
          setMaxLat(round5(bMaxLat));
          setMaxLon(round5(bMaxLon));
          setActiveTab("coordinates");
        } else {
          setValidationError("Could not extract valid polygon coordinates from GeoJSON.");
        }
      } catch (err) {
        setValidationError("Invalid JSON/GeoJSON format: " + err.message);
      }
    };
    reader.readAsText(file);
  };

  const round5 = (num) => Math.round(num * 100000) / 100000;

  const handleApply = () => {
    if (!previewAOI && !validationError) return;
    onApplyAOI({
      name: manualName,
      bbox: [Number(minLat), Number(minLon), Number(maxLat), Number(maxLon)],
      preset_id: previewAOI?.id?.includes("aoi_") ? null : previewAOI?.preset_id,
      terrain_type: terrainType,
      area_km2: previewAOI?.area_km2,
      utm_crs: previewAOI?.utm_crs,
    });
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-md flex items-center justify-center p-4 overflow-y-auto">
      <div className="w-full max-w-3xl bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="px-5 py-3.5 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-cyan-500/15 border border-cyan-500/40 text-cyan-400">
              <Compass className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-mono font-bold text-slate-100">
                  UNIVERSAL STUDY AREA &amp; AOI SELECTOR
                </h2>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                  GEOGRAPHICALLY CONFIGURABLE
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Analyze any city, river basin, coastal estuary, or watershed worldwide.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-100 border border-slate-800 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="px-5 pt-3 bg-slate-950 border-b border-slate-800 flex gap-2 overflow-x-auto">
          {[
            { id: "search", label: "Search Location", icon: Search },
            { id: "coordinates", label: "Coordinates & Bounds", icon: Sliders },
            { id: "upload", label: "Upload Boundary", icon: Upload },
            { id: "presets", label: "Saved Templates", icon: Bookmark },
          ].map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-2 text-xs font-mono font-semibold border-b-2 transition whitespace-nowrap ${
                  activeTab === tab.id
                    ? "border-cyan-400 text-cyan-300 bg-cyan-500/10 rounded-t"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab Content Area */}
        <div className="p-5 overflow-y-auto space-y-4 flex-1">
          {/* TAB 1: SEARCH LOCATION */}
          {activeTab === "search" && (
            <div className="space-y-3">
              <form onSubmit={handleSearch} className="flex gap-2">
                <div className="relative flex-1">
                  <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search any location (e.g. Cuddalore, Patna, Houston, Venice, Kaziranga)..."
                    className="w-full pl-9 pr-3 py-2 bg-slate-950 border border-slate-700 rounded-lg text-xs font-mono text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400"
                  />
                </div>
                <button
                  type="submit"
                  disabled={isSearching || !searchQuery.trim()}
                  className="px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-mono font-bold rounded-lg transition disabled:opacity-40 flex items-center gap-1.5"
                >
                  {isSearching ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : "Search"}
                </button>
              </form>

              {searchError && (
                <div className="p-2.5 rounded bg-rose-950/40 border border-rose-500/40 text-xs font-mono text-rose-300 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
                  <span>{searchError}</span>
                </div>
              )}

              {/* Search Results List */}
              <div className="space-y-2 max-h-56 overflow-y-auto">
                {searchResults.map((loc, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-lg bg-slate-950 border border-slate-800 hover:border-cyan-500/50 transition flex items-center justify-between gap-3"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <MapPin className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                        <span className="text-xs font-semibold text-slate-200 truncate">{loc.name}</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-400">
                          {loc.type}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 font-mono truncate mt-0.5">{loc.display_name}</p>
                      <div className="text-[10px] font-mono text-slate-500 mt-1">
                        BBox: [{loc.bbox.join(", ")}] | Area: ~{loc.area_km2.toFixed(1)} km²
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleSelectSearchResult(loc)}
                      className="px-3 py-1.5 rounded bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-300 text-xs font-mono font-semibold border border-cyan-500/40 shrink-0"
                    >
                      Select AOI
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 2: MANUAL COORDINATES */}
          {activeTab === "coordinates" && (
            <div className="space-y-3">
              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">Study Area Label / Sector Name:</label>
                <input
                  type="text"
                  value={manualName}
                  onChange={(e) => setManualName(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-lg text-xs font-mono text-slate-100"
                />
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                <div>
                  <label className="text-[11px] font-mono text-slate-400 block mb-1">Min Latitude (°N):</label>
                  <input
                    type="number"
                    step="0.001"
                    value={minLat}
                    onChange={(e) => setMinLat(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 bg-slate-950 border border-slate-700 rounded text-xs font-mono text-slate-100"
                  />
                </div>
                <div>
                  <label className="text-[11px] font-mono text-slate-400 block mb-1">Min Longitude (°E):</label>
                  <input
                    type="number"
                    step="0.001"
                    value={minLon}
                    onChange={(e) => setMinLon(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 bg-slate-950 border border-slate-700 rounded text-xs font-mono text-slate-100"
                  />
                </div>
                <div>
                  <label className="text-[11px] font-mono text-slate-400 block mb-1">Max Latitude (°N):</label>
                  <input
                    type="number"
                    step="0.001"
                    value={maxLat}
                    onChange={(e) => setMaxLat(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 bg-slate-950 border border-slate-700 rounded text-xs font-mono text-slate-100"
                  />
                </div>
                <div>
                  <label className="text-[11px] font-mono text-slate-400 block mb-1">Max Longitude (°E):</label>
                  <input
                    type="number"
                    step="0.001"
                    value={maxLon}
                    onChange={(e) => setMaxLon(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 bg-slate-950 border border-slate-700 rounded text-xs font-mono text-slate-100"
                  />
                </div>
              </div>

              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">Hydrological Terrain Profile:</label>
                <select
                  value={terrainType}
                  onChange={(e) => setTerrainType(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-lg text-xs font-mono text-slate-100"
                >
                  <option value="auto">Auto-Detect from Coordinates &amp; Elevation</option>
                  <option value="coastal_urban">Coastal Urban &amp; Estuary (Bay of Bengal / Coast)</option>
                  <option value="river_floodplain">River Floodplain (Alluvial Basin, e.g. Brahmaputra)</option>
                  <option value="mountainous_catchment">Mountainous Highland Catchment (Western Ghats, Steep Slopes)</option>
                  <option value="urban_drainage_basin">Urban Drainage Basin (Inland City Arterial Corridor)</option>
                </select>
              </div>
            </div>
          )}

          {/* TAB 3: UPLOAD BOUNDARY FILE */}
          {activeTab === "upload" && (
            <div className="p-6 border-2 border-dashed border-slate-700 hover:border-cyan-400/60 rounded-xl text-center space-y-3 transition">
              <Upload className="w-8 h-8 text-cyan-400 mx-auto" />
              <div>
                <div className="text-xs font-mono font-semibold text-slate-200">
                  Upload GeoJSON Study Area Boundary File
                </div>
                <p className="text-[11px] font-mono text-slate-400 mt-1">
                  Supports standard GeoJSON FeatureCollection or Polygon geometries (.geojson, .json).
                </p>
              </div>
              <label className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-300 text-xs font-mono font-semibold border border-cyan-500/40 cursor-pointer transition">
                Browse GeoJSON File
                <input
                  type="file"
                  accept=".geojson,.json"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>
          )}

          {/* TAB 4: SAVED DEMO TEMPLATES */}
          {activeTab === "presets" && (
            <div className="space-y-2">
              <p className="text-xs font-mono text-slate-400">
                Pre-configured study-area case studies for instant viva demonstrations:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                {(presets || []).map((p) => (
                  <div
                    key={p.id}
                    className="p-3 rounded-lg bg-slate-950 border border-slate-800 hover:border-cyan-400/50 transition cursor-pointer flex flex-col justify-between"
                    onClick={() => handleSelectPreset(p)}
                  >
                    <div>
                      <div className="text-xs font-semibold text-slate-200 mb-1">{p.name}</div>
                      <div className="text-[10px] font-mono text-slate-400">{p.region}</div>
                    </div>
                    <button
                      type="button"
                      className="mt-3 w-full py-1 text-[11px] font-mono font-semibold bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 rounded hover:bg-cyan-500/20"
                    >
                      Load Template
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* LIVE VALIDATION & METRIC PREVIEW BOX */}
          <div className="p-3.5 rounded-lg bg-slate-950/90 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                <Globe className="w-3.5 h-3.5 text-cyan-400" />
                AOI Geodetic Telemetry Preview:
              </span>
              {isValidating && <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />}
            </div>

            {validationError ? (
              <div className="p-2 rounded bg-rose-950/40 border border-rose-500/40 text-[11px] font-mono text-rose-300 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{validationError}</span>
              </div>
            ) : previewAOI ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px]">
                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Surface Area</div>
                  <div className="text-sm font-bold text-cyan-300">
                    {previewAOI.area_km2?.toFixed(2)} km²
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Local Projection</div>
                  <div className="text-sm font-bold text-emerald-400">
                    {previewAOI.utm_crs || "EPSG:32644"}
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Terrain Profile</div>
                  <div className="text-xs font-semibold text-amber-300 truncate">
                    {(previewAOI.terrain_type || terrainType).replace("_", " ").toUpperCase()}
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Processing Load</div>
                  <div className="text-xs font-semibold text-slate-200">
                    {previewAOI.area_km2 > 120 ? (
                      <span className="text-amber-400">Multi-Tile Stitched</span>
                    ) : (
                      <span className="text-emerald-400">Optimal (10m Res)</span>
                    )}
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-5 py-3.5 bg-slate-950 border-t border-slate-800 flex items-center justify-between">
          <button
            type="button"
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs font-mono font-semibold border border-slate-800 transition"
          >
            Cancel
          </button>

          <button
            type="button"
            onClick={handleApply}
            disabled={Boolean(validationError) || isValidating}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-mono font-bold transition disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_0_12px_rgba(6,182,212,0.35)]"
          >
            <CheckCircle2 className="w-4 h-4" />
            Apply Study Area &amp; Analyze
          </button>
        </div>
      </div>
    </div>
  );
}
