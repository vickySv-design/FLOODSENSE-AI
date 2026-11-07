import React, { useState } from "react";
import {
  CloudRain,
  Compass,
  FolderUp,
  Globe,
  Layers,
  Loader2,
  MapPin,
  Play,
  Radio,
  Sparkles,
} from "lucide-react";

export default function ControlSidebar({
  presets,
  selectedPresetId,
  onSelectPreset,
  onUploadCustomPair,
  onFetchGEE,
  layerVisibility,
  onToggleLayer,
  isLoading,
  onOpenAOIModal,
}) {
  const [activeTab, setActiveTab] = useState("presets");

  // Custom Upload state
  const [preOpticalFile, setPreOpticalFile] = useState(null);
  const [postSarFile, setPostSarFile] = useState(null);
  const [customMissionName, setCustomMissionName] = useState(
    "Custom Sentinel-1/2 Mission"
  );

  // Google Earth Engine (GEE) state
  const [geeForm, setGeeForm] = useState({
    region_name: "Adyar Estuary & Coastal Corridor",
    min_lat: 13.01,
    min_lon: 80.22,
    max_lat: 13.06,
    max_lon: 80.27,
    pre_date: "2025-11-10",
    post_date: "2025-12-05",
    cloud_cover_max: 20,
  });

  const handleCustomSubmit = (e) => {
    e.preventDefault();
    if (!preOpticalFile || !postSarFile) return;
    onUploadCustomPair({
      preOpticalFile,
      postSarFile,
      customName: customMissionName,
    });
  };

  const handleGeeSubmit = (e) => {
    e.preventDefault();
    onFetchGEE({
      ...geeForm,
      min_lat: parseFloat(geeForm.min_lat),
      min_lon: parseFloat(geeForm.min_lon),
      max_lat: parseFloat(geeForm.max_lat),
      max_lon: parseFloat(geeForm.max_lon),
      cloud_cover_max: parseFloat(geeForm.cloud_cover_max),
    });
  };

  const layerItems = [
    {
      key: "allRoads",
      label: "OpenStreetMap Road Network",
      color: "bg-slate-400",
      badge: "Real OSM",
    },
    {
      key: "newFlood",
      label: "Flood Inundation Polygons",
      color: "bg-rose-500",
      badge: "Class 2",
    },
    {
      key: "permanentWater",
      label: "Permanent Water Baseline",
      color: "bg-cyan-400",
      badge: "Class 1",
    },
    {
      key: "highRiskBuffer",
      label: "100m High-Risk Buffer",
      color: "bg-amber-400",
      badge: "Dilation",
    },
    {
      key: "buildings",
      label: "Building Damage Footprints",
      color: "bg-emerald-400",
      badge: "OSM Polygons",
    },
    {
      key: "safeRoutes",
      label: "Safe A* Rescue Routes",
      color: "bg-emerald-500",
      badge: "OSM A*",
    },
    {
      key: "severedRoads",
      label: "Severed / Flooded Roads",
      color: "bg-red-500",
      badge: "Blocked OSM",
    },
  ];

  return (
    <aside className="w-full lg:w-80 xl:w-88 shrink-0 bg-slate-900/90 border-r border-slate-800 flex flex-col h-full overflow-y-auto select-none">
      {/* Universal Study Area (AOI) Selector Card */}
      <div className="p-3 border-b border-slate-800 bg-slate-950/60">
        <div className="flex items-center justify-between text-[11px] font-mono mb-2">
          <span className="text-slate-300 font-bold flex items-center gap-1.5 uppercase tracking-wider">
            <Compass className="w-3.5 h-3.5 text-cyan-400" />
            Study Area (AOI)
          </span>
          <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20 font-mono">
            Configurable
          </span>
        </div>
        <button
          type="button"
          onClick={onOpenAOIModal}
          className="w-full py-2 px-3 rounded-lg bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-300 border border-cyan-500/40 text-xs font-mono font-bold transition flex items-center justify-center gap-1.5 shadow-[0_0_8px_rgba(6,182,212,0.2)]"
        >
          <Globe className="w-3.5 h-3.5" />
          Configure Any Study Area
        </button>
      </div>

      {/* Mode Selector Tabs */}
      <div className="p-3 border-b border-slate-800">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
          <Radio className="w-3.5 h-3.5 text-cyan-400" />
          Data Acquisition Source
        </div>
        <div className="grid grid-cols-3 gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
          <button
            type="button"
            onClick={() => setActiveTab("presets")}
            className={`flex items-center justify-center gap-1 py-1.5 px-2 rounded-md text-xs font-medium transition ${
              activeTab === "presets"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            Presets
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("upload")}
            className={`flex items-center justify-center gap-1 py-1.5 px-2 rounded-md text-xs font-medium transition ${
              activeTab === "upload"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <FolderUp className="w-3.5 h-3.5" />
            Upload
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("gee")}
            className={`flex items-center justify-center gap-1 py-1.5 px-2 rounded-md text-xs font-medium transition ${
              activeTab === "gee"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Globe className="w-3.5 h-3.5" />
            GEE Live
          </button>
        </div>
      </div>

      {/* Tab Content Area */}
      <div className="p-3.5 border-b border-slate-800 flex-1">
        {activeTab === "presets" && (
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">
                Pre-Loaded Disaster Case Studies
              </span>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-1.5 py-0.5 rounded">
                3 Verified AOIs
              </span>
            </div>

            {presets.map((preset) => {
              const isSelected = preset.id === selectedPresetId;
              return (
                <button
                  key={preset.id}
                  type="button"
                  disabled={isLoading}
                  onClick={() => onSelectPreset(preset.id)}
                  className={`w-full text-left p-3 rounded-lg border transition relative overflow-hidden ${
                    isSelected
                      ? "bg-cyan-950/40 border-cyan-400/60 shadow-lg shadow-cyan-950/50"
                      : "bg-slate-950/70 border-slate-800/90 hover:border-slate-700"
                  }`}
                >
                  {isSelected && (
                    <span className="absolute top-0 left-0 bottom-0 w-1 bg-cyan-400" />
                  )}
                  <div className="flex items-start justify-between gap-2">
                    <h4 className="text-xs font-bold text-slate-100 leading-snug">
                      {preset.name}
                    </h4>
                    <span className="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 text-cyan-300 border border-slate-700">
                      {preset.event_date}
                    </span>
                  </div>

                  <div className="flex items-center gap-1 text-[11px] text-slate-400 mt-1">
                    <MapPin className="w-3 h-3 text-rose-400 shrink-0" />
                    <span className="truncate">{preset.region}</span>
                  </div>

                  <p className="text-[11px] text-slate-400 mt-1.5 line-clamp-2 leading-relaxed">
                    {preset.description}
                  </p>

                  <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-400">
                    <span>
                      BBox: [{preset.bbox[0].toFixed(2)}°N,{" "}
                      {preset.bbox[1].toFixed(2)}°E]
                    </span>
                    <span className="text-emerald-400 font-semibold">
                      mIoU {(preset.historical_metrics?.miou * 100).toFixed(1)}%
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}

        {activeTab === "upload" && (
          <form onSubmit={handleCustomSubmit} className="space-y-3">
            <div className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
              Multi-Modal Satellite Pair Upload
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Upload co-registered Pre-Event Optical (Sentinel-2 RGB/NIR or
              .npy) and Post-Event SAR (Sentinel-1 VV/VH or .npy) tiles for live
              Siamese Attention U-Net inference.
            </p>

            <div>
              <label className="block text-[11px] font-mono text-slate-300 mb-1">
                Mission Designation
              </label>
              <input
                type="text"
                value={customMissionName}
                onChange={(e) => setCustomMissionName(e.target.value)}
                className="w-full px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div>
              <label className="block text-[11px] font-mono text-cyan-300 mb-1">
                1. Pre-Event Optical Tile (Sentinel-2)
              </label>
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff,.npy"
                onChange={(e) => setPreOpticalFile(e.target.files?.[0] || null)}
                className="w-full text-xs text-slate-300 file:mr-2 file:py-1.5 file:px-2.5 file:rounded file:border-0 file:text-xs file:font-mono file:bg-cyan-500/20 file:text-cyan-300 hover:file:bg-cyan-500/30 bg-slate-950 border border-slate-800 rounded p-1"
              />
            </div>

            <div>
              <label className="block text-[11px] font-mono text-rose-300 mb-1">
                2. Post-Event SAR Tile (Sentinel-1 VV/VH)
              </label>
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff,.npy"
                onChange={(e) => setPostSarFile(e.target.files?.[0] || null)}
                className="w-full text-xs text-slate-300 file:mr-2 file:py-1.5 file:px-2.5 file:rounded file:border-0 file:text-xs file:font-mono file:bg-rose-500/20 file:text-rose-300 hover:file:bg-rose-500/30 bg-slate-950 border border-slate-800 rounded p-1"
              />
            </div>

            <button
              type="submit"
              disabled={!preOpticalFile || !postSarFile || isLoading}
              className="w-full py-2 px-3 rounded-md bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-mono text-xs font-bold flex items-center justify-center gap-1.5 transition disabled:opacity-40"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              Execute Dual-Stream Inference
            </button>
          </form>
        )}

        {activeTab === "gee" && (
          <form onSubmit={handleGeeSubmit} className="space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <Globe className="w-3.5 h-3.5 text-emerald-400" />
                Google Earth Engine Fetch
              </span>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                S1 + S2 SR
              </span>
            </div>

            <div>
              <label className="block text-[10px] font-mono text-slate-400 mb-0.5">
                Target Sector Name
              </label>
              <input
                type="text"
                value={geeForm.region_name}
                onChange={(e) =>
                  setGeeForm({ ...geeForm, region_name: e.target.value })
                }
                className="w-full px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs text-slate-100"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] font-mono text-slate-400 mb-0.5">
                  Min Lat (°N)
                </label>
                <input
                  type="number"
                  step="0.005"
                  value={geeForm.min_lat}
                  onChange={(e) =>
                    setGeeForm({ ...geeForm, min_lat: e.target.value })
                  }
                  className="w-full px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-100"
                />
              </div>
              <div>
                <label className="block text-[10px] font-mono text-slate-400 mb-0.5">
                  Max Lat (°N)
                </label>
                <input
                  type="number"
                  step="0.005"
                  value={geeForm.max_lat}
                  onChange={(e) =>
                    setGeeForm({ ...geeForm, max_lat: e.target.value })
                  }
                  className="w-full px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-100"
                />
              </div>
              <div>
                <label className="block text-[10px] font-mono text-slate-400 mb-0.5">
                  Min Lon (°E)
                </label>
                <input
                  type="number"
                  step="0.005"
                  value={geeForm.min_lon}
                  onChange={(e) =>
                    setGeeForm({ ...geeForm, min_lon: e.target.value })
                  }
                  className="w-full px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-100"
                />
              </div>
              <div>
                <label className="block text-[10px] font-mono text-slate-400 mb-0.5">
                  Max Lon (°E)
                </label>
                <input
                  type="number"
                  step="0.005"
                  value={geeForm.max_lon}
                  onChange={(e) =>
                    setGeeForm({ ...geeForm, max_lon: e.target.value })
                  }
                  className="w-full px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-100"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] font-mono text-slate-400 mb-0.5">
                  Pre-Flood Date
                </label>
                <input
                  type="date"
                  value={geeForm.pre_date}
                  onChange={(e) =>
                    setGeeForm({ ...geeForm, pre_date: e.target.value })
                  }
                  className="w-full px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-100"
                />
              </div>
              <div>
                <label className="block text-[10px] font-mono text-slate-400 mb-0.5">
                  Post-Flood SAR Date
                </label>
                <input
                  type="date"
                  value={geeForm.post_date}
                  onChange={(e) =>
                    setGeeForm({ ...geeForm, post_date: e.target.value })
                  }
                  className="w-full px-2 py-1 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-100"
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-[10px] font-mono text-slate-400 mb-1">
                <span className="flex items-center gap-1">
                  <CloudRain className="w-3 h-3 text-cyan-400" />
                  Max Optical Cloud Cover
                </span>
                <span className="text-cyan-300">{geeForm.cloud_cover_max}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="80"
                value={geeForm.cloud_cover_max}
                onChange={(e) =>
                  setGeeForm({ ...geeForm, cloud_cover_max: e.target.value })
                }
                className="w-full accent-cyan-400 cursor-pointer"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-2 px-3 rounded-md bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-mono text-xs font-bold flex items-center justify-center gap-1.5 transition disabled:opacity-40"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Globe className="w-4 h-4" />
              )}
              Fetch Sentinel-1/2 Tiles
            </button>
          </form>
        )}
      </div>

      {/* Interactive Layer Visibility Switches */}
      <div className="p-3.5 bg-slate-950/60">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-2.5 flex items-center justify-between">
          <span className="flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            Tactical GIS Layer Toggles
          </span>
          <span className="text-[10px] text-slate-500">Vector Overlay</span>
        </div>

        <div className="space-y-1.5">
          {layerItems.map((item) => {
            const active = Boolean(layerVisibility[item.key]);
            return (
              <label
                key={item.key}
                className="flex items-center justify-between p-2 rounded-md bg-slate-900/90 border border-slate-800/80 hover:border-slate-700 cursor-pointer transition"
              >
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${item.color}`} />
                  <span className="text-xs text-slate-200">{item.label}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-slate-400 px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800">
                    {item.badge}
                  </span>
                  <input
                    type="checkbox"
                    checked={active}
                    onChange={() => onToggleLayer(item.key)}
                    className="w-3.5 h-3.5 accent-cyan-400 rounded cursor-pointer"
                  />
                </div>
              </label>
            );
          })}
        </div>
      </div>
    </aside>
  );
}

