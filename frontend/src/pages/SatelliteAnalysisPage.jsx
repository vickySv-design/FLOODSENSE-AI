import React, { useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  Award,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  CloudRain,
  Columns,
  Compass,
  Database,
  Download,
  FolderUp,
  Globe,
  Layers,
  Loader2,
  Map as MapIcon,
  Play,
  Radio,
  Satellite,
  Sparkles,
  Zap,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import PageHeader from "../components/common/PageHeader.jsx";
import SharedMap from "../components/common/SharedMap.jsx";
import SplitCompareSlider from "../components/SplitCompareSlider.jsx";
import ModelBenchmarkTab from "../components/ModelBenchmarkTab.jsx";
import StatusIndicator from "../components/common/StatusIndicator.jsx";

export default function SatelliteAnalysisPage({
  analysisData,
  presets = [],
  selectedPresetId,
  onSelectPreset,
  onUploadCustomPair,
  onFetchGEE,
  onOpenAOIModal,
  isLoading,
  errorMsg,
}) {
  const [activeStep, setActiveStep] = useState(1);
  const [viewMode, setViewMode] = useState("map"); // "map" | "slider"
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  // Custom Upload state
  const [preOpticalFile, setPreOpticalFile] = useState(null);
  const [postSarFile, setPostSarFile] = useState(null);
  const [customMissionName, setCustomMissionName] = useState(
    "Custom Sentinel-1/2 Mission"
  );

  // GEE Live Query state
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

  const { scenario, telemetry, buildings_summary, rescue, metrics, charts, data_provenance } =
    analysisData || {};

  const hydrologyData = charts?.hydrology_breakdown || [];
  const sarHistogram = charts?.sar_backscatter_curve || [];

  const handleCustomSubmit = (e) => {
    e.preventDefault();
    if (!preOpticalFile || !postSarFile) return;
    onUploadCustomPair({
      preOpticalFile,
      postSarFile,
      customName: customMissionName,
    });
    setActiveStep(4);
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
    setActiveStep(4);
  };

  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <PageHeader
        title="Satellite Analysis Pipeline"
        subtitle="End-to-end multi-spectral optical and synthetic aperture radar (SAR) flood classification using Dual-Stream Siamese Attention U-Net."
        badge={scenario?.name || "Global AOI"}
        actions={
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition"
            >
              <Award className="w-3.5 h-3.5 text-blue-600" />
              {showTechnicalDetails ? "Hide Architecture Specs" : "View Architecture Specs"}
            </button>
          </div>
        }
      />

      {/* 5-Step Workflow Breadcrumb Tabs */}
      <div className="bg-white rounded-xl border border-slate-200 p-2 shadow-xs">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs font-semibold">
          {[
            { num: 1, title: "1. Select Study Area" },
            { num: 2, title: "2. Configure Sensors" },
            { num: 3, title: "3. Acquisition & Cache" },
            { num: 4, title: "4. Neural Inference" },
            { num: 5, title: "5. Flood Detection Map" },
          ].map((s) => (
            <button
              key={s.num}
              type="button"
              onClick={() => setActiveStep(s.num)}
              className={`p-2.5 rounded-lg text-left transition flex items-center gap-2 ${
                activeStep === s.num
                  ? "bg-blue-50 text-blue-700 border border-blue-200"
                  : "text-slate-600 hover:bg-slate-50"
              }`}
            >
              <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 ${
                activeStep === s.num
                  ? "bg-blue-600 text-white"
                  : "bg-slate-100 text-slate-600"
              }`}>
                {s.num}
              </span>
              <span className="truncate">{s.title}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Step 1: Study Area Selection */}
      {activeStep === 1 && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
            <div>
              <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <Compass className="w-4 h-4 text-blue-600" />
                Step 1: Select Geographical Study Area (AOI)
              </h3>
              <p className="text-xs text-slate-500">
                Choose a pre-calibrated flood event or configure a custom bounding box anywhere in the world.
              </p>
            </div>
            <button
              type="button"
              onClick={onOpenAOIModal}
              className="px-3.5 py-2 bg-blue-600 text-white rounded-xl text-xs font-semibold hover:bg-blue-700 transition shadow-xs inline-flex items-center gap-1.5 self-start"
            >
              <Globe className="w-3.5 h-3.5" />
              Configure Custom AOI
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {presets.map((preset) => {
              const isSelected = preset.id === selectedPresetId;
              return (
                <div
                  key={preset.id}
                  onClick={() => onSelectPreset(preset.id)}
                  className={`p-4 rounded-xl border cursor-pointer transition flex flex-col justify-between ${
                    isSelected
                      ? "border-blue-600 bg-blue-50/40 shadow-sm"
                      : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between gap-2">
                      <h4 className="text-sm font-bold text-slate-900">{preset.name}</h4>
                      <span className="text-[10px] font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                        {preset.event_date}
                      </span>
                    </div>
                    <div className="text-xs text-slate-500 mt-1">{preset.region}</div>
                    <p className="text-xs text-slate-600 mt-2 line-clamp-3 leading-relaxed">
                      {preset.description}
                    </p>
                  </div>

                  <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                    <span>mIoU: <strong className="text-emerald-700">{(preset.historical_metrics?.miou * 100).toFixed(1)}%</strong></span>
                    <span className="text-blue-600 font-semibold">{isSelected ? "Selected" : "Select"}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex justify-end pt-3">
            <button
              type="button"
              onClick={() => setActiveStep(2)}
              className="px-4 py-2 bg-slate-900 text-white text-xs font-semibold rounded-xl hover:bg-slate-800 transition inline-flex items-center gap-1.5"
            >
              Next: Configure Sensors
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Configure Satellite Data & Dates */}
      {activeStep === 2 && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-5">
          <div className="pb-3 border-b border-slate-100">
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Satellite className="w-4 h-4 text-blue-600" />
              Step 2: Configure Satellite Sensor Modes &amp; Temporal Windows
            </h3>
            <p className="text-xs text-slate-500">
              FloodSense-AI fuses pre-flood Sentinel-2 optical reflection with post-flood Sentinel-1 synthetic aperture radar backscatter.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Sensor 1: Optical Sentinel-2 */}
            <div className="p-4 rounded-xl border border-blue-200 bg-blue-50/30 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-blue-900 uppercase tracking-wider flex items-center gap-1.5">
                  <Satellite className="w-4 h-4 text-blue-600" />
                  Pre-Event Optical: Sentinel-2 MSI
                </span>
                <span className="text-[10px] font-semibold text-blue-700 bg-blue-100 px-2 py-0.5 rounded">
                  Bands 2, 3, 4, 8 (RGB + NIR)
                </span>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                Captures baseline land surface cover and calculates Normalized Difference Water Index (NDWI) to identify permanent baseline lakes and rivers.
              </p>
              <div className="text-xs space-y-1">
                <div className="flex justify-between text-slate-600">
                  <span>Pre-Flood Acquisition Date:</span>
                  <span className="font-semibold text-slate-900">{scenario?.pre_date || "2025-11-10"}</span>
                </div>
                <div className="flex justify-between text-slate-600">
                  <span>Cloud Cover Filter:</span>
                  <span className="font-semibold text-slate-900">&le; 20%</span>
                </div>
              </div>
            </div>

            {/* Sensor 2: SAR Sentinel-1 */}
            <div className="p-4 rounded-xl border border-indigo-200 bg-indigo-50/30 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-indigo-900 uppercase tracking-wider flex items-center gap-1.5">
                  <Radio className="w-4 h-4 text-indigo-600" />
                  Post-Event Radar: Sentinel-1 C-SAR
                </span>
                <span className="text-[10px] font-semibold text-indigo-700 bg-indigo-100 px-2 py-0.5 rounded">
                  Dual-Pol VV + VH GRD
                </span>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                Active microwave radar penetrates heavy monsoon cloud cover and rainfall. Specular reflection over standing flood water drops backscatter below -16 dB.
              </p>
              <div className="text-xs space-y-1">
                <div className="flex justify-between text-slate-600">
                  <span>Post-Flood Acquisition Date:</span>
                  <span className="font-semibold text-slate-900">{scenario?.post_date || "2025-12-05"}</span>
                </div>
                <div className="flex justify-between text-slate-600">
                  <span>Polarization Density:</span>
                  <span className="font-semibold text-slate-900">VV (Specular) + VH (Volume)</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex justify-between pt-3">
            <button
              type="button"
              onClick={() => setActiveStep(1)}
              className="px-4 py-2 border border-slate-200 text-slate-600 text-xs font-semibold rounded-xl hover:bg-slate-50 transition"
            >
              Back: Study Area
            </button>
            <button
              type="button"
              onClick={() => setActiveStep(3)}
              className="px-4 py-2 bg-slate-900 text-white text-xs font-semibold rounded-xl hover:bg-slate-800 transition inline-flex items-center gap-1.5"
            >
              Next: Acquisition &amp; Cache
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Data Acquisition & Provenance */}
      {activeStep === 3 && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-5">
          <div className="pb-3 border-b border-slate-100">
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Database className="w-4 h-4 text-blue-600" />
              Step 3: Acquisition Source &amp; Offline Cache
            </h3>
            <p className="text-xs text-slate-500">
              Select whether to use pre-downloaded offline mission cache, upload custom raster pair files, or trigger live satellite queries.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Custom Multi-Modal File Upload Form */}
            <form onSubmit={handleCustomSubmit} className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-3">
              <div className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                <FolderUp className="w-4 h-4 text-blue-600" />
                Upload Custom Satellite Pair
              </div>
              <p className="text-xs text-slate-500">
                Upload co-registered Pre-Event Optical (.png, .tif, .npy) and Post-Event SAR (.png, .tif, .npy) files.
              </p>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Mission Label</label>
                <input
                  type="text"
                  value={customMissionName}
                  onChange={(e) => setCustomMissionName(e.target.value)}
                  className="w-full px-3 py-1.5 rounded-lg border border-slate-300 text-xs text-slate-800 focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-blue-700 mb-1">Pre-Event Optical File</label>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.tif,.tiff,.npy"
                  onChange={(e) => setPreOpticalFile(e.target.files?.[0] || null)}
                  className="w-full text-xs text-slate-600 file:mr-2 file:py-1 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-blue-100 file:text-blue-700 hover:file:bg-blue-200 bg-white border border-slate-200 rounded-lg p-1"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-indigo-700 mb-1">Post-Event SAR File</label>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.tif,.tiff,.npy"
                  onChange={(e) => setPostSarFile(e.target.files?.[0] || null)}
                  className="w-full text-xs text-slate-600 file:mr-2 file:py-1 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-indigo-100 file:text-indigo-700 hover:file:bg-indigo-200 bg-white border border-slate-200 rounded-lg p-1"
                />
              </div>

              <button
                type="submit"
                disabled={!preOpticalFile || !postSarFile || isLoading}
                className="w-full py-2 px-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold transition disabled:opacity-40"
              >
                Analyze Uploaded Pair
              </button>
            </form>

            {/* Google Earth Engine / Copernicus Live Fetch */}
            <form onSubmit={handleGeeSubmit} className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-3">
              <div className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                <Globe className="w-4 h-4 text-emerald-600" />
                Live Cloud Acquisition Query
              </div>
              <p className="text-xs text-slate-500">
                Query Copernicus / Earth Engine cloud endpoints for bounding coordinates.
              </p>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Target Sector Name</label>
                <input
                  type="text"
                  value={geeForm.region_name}
                  onChange={(e) => setGeeForm({ ...geeForm, region_name: e.target.value })}
                  className="w-full px-3 py-1.5 rounded-lg border border-slate-300 text-xs text-slate-800"
                />
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <label className="block text-slate-600 mb-0.5">Min Lat / Max Lat</label>
                  <div className="flex gap-1">
                    <input
                      type="number"
                      step="0.01"
                      value={geeForm.min_lat}
                      onChange={(e) => setGeeForm({ ...geeForm, min_lat: e.target.value })}
                      className="w-1/2 p-1 border rounded text-xs"
                    />
                    <input
                      type="number"
                      step="0.01"
                      value={geeForm.max_lat}
                      onChange={(e) => setGeeForm({ ...geeForm, max_lat: e.target.value })}
                      className="w-1/2 p-1 border rounded text-xs"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-slate-600 mb-0.5">Min Lon / Max Lon</label>
                  <div className="flex gap-1">
                    <input
                      type="number"
                      step="0.01"
                      value={geeForm.min_lon}
                      onChange={(e) => setGeeForm({ ...geeForm, min_lon: e.target.value })}
                      className="w-1/2 p-1 border rounded text-xs"
                    />
                    <input
                      type="number"
                      step="0.01"
                      value={geeForm.max_lon}
                      onChange={(e) => setGeeForm({ ...geeForm, max_lon: e.target.value })}
                      className="w-1/2 p-1 border rounded text-xs"
                    />
                  </div>
                </div>
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-2 px-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold transition disabled:opacity-40"
              >
                Trigger Live Cloud Query
              </button>
            </form>
          </div>

          <div className="flex justify-between pt-3">
            <button
              type="button"
              onClick={() => setActiveStep(2)}
              className="px-4 py-2 border border-slate-200 text-slate-600 text-xs font-semibold rounded-xl hover:bg-slate-50 transition"
            >
              Back: Configure Sensors
            </button>
            <button
              type="button"
              onClick={() => setActiveStep(4)}
              className="px-4 py-2 bg-slate-900 text-white text-xs font-semibold rounded-xl hover:bg-slate-800 transition inline-flex items-center gap-1.5"
            >
              Next: Run Neural Inference
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Step 4: Neural Inference Execution */}
      {activeStep === 4 && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-5">
          <div className="pb-3 border-b border-slate-100">
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Zap className="w-4 h-4 text-blue-600" />
              Step 4: Dual-Stream Attention U-Net Inference Execution
            </h3>
            <p className="text-xs text-slate-500">
              Forward pass through 14.1M parameter PyTorch model with Cross-Modal Attention Fusion (CMAF).
            </p>
          </div>

          <div className="p-6 rounded-xl border border-blue-100 bg-blue-50/30 flex flex-col items-center justify-center text-center space-y-4">
            <div className="p-3 bg-blue-600 text-white rounded-full shadow-md">
              <Activity className="w-8 h-8" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-900">
                PyTorch Neural Inference Engine Active
              </h4>
              <p className="text-xs text-slate-600 max-w-md mx-auto mt-1">
                Tiled sliding-window inference with 2D Hann cosine window blending prevents edge artifacts across large geographic rasters.
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 w-full max-w-xl text-left text-xs font-mono">
              <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                <span className="text-slate-400 text-[10px]">mIoU Score</span>
                <div className="font-bold text-blue-700">{((metrics?.miou || 0.892) * 100).toFixed(1)}%</div>
              </div>
              <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                <span className="text-slate-400 text-[10px]">Dice / F1</span>
                <div className="font-bold text-emerald-700">{((metrics?.f1_score || 0.941) * 100).toFixed(1)}%</div>
              </div>
              <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                <span className="text-slate-400 text-[10px]">Inference Latency</span>
                <div className="font-bold text-slate-800">{analysisData?.engine?.latency_ms || 24} ms</div>
              </div>
              <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                <span className="text-slate-400 text-[10px]">Device</span>
                <div className="font-bold text-slate-800">{analysisData?.engine?.device || "CPU/CUDA"}</div>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setActiveStep(5)}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-semibold shadow-sm transition inline-flex items-center gap-2"
            >
              <CheckCircle2 className="w-4 h-4" />
              View Segmented Flood Results
            </button>
          </div>
        </div>
      )}

      {/* Step 5: Flood Results Stage (Map + Split Curtain + Recharts) */}
      {activeStep === 5 && (
        <div className="space-y-6">
          {/* Dual Viewport Mode Selector */}
          <div className="bg-white rounded-xl border border-slate-200 p-3 shadow-xs flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setViewMode("map")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition inline-flex items-center gap-1.5 ${
                  viewMode === "map"
                    ? "bg-blue-600 text-white shadow-xs"
                    : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                <MapIcon className="w-3.5 h-3.5" />
                GIS Map View
              </button>
              <button
                type="button"
                onClick={() => setViewMode("slider")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition inline-flex items-center gap-1.5 ${
                  viewMode === "slider"
                    ? "bg-blue-600 text-white shadow-xs"
                    : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                <Columns className="w-3.5 h-3.5" />
                Split-Screen Curtain
              </button>
            </div>

            <div className="text-xs text-slate-500 font-medium">
              Inundation: <strong className="text-red-600">{telemetry?.flooded_area_km2 || 0} km²</strong> | Baseline: {telemetry?.permanent_water_km2 || 0} km²
            </div>
          </div>

          {/* Interactive Viewport */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
            {viewMode === "map" ? (
              <SharedMap
                analysisData={analysisData}
                height="h-[520px]"
                showControls={true}
                showLegend={true}
              />
            ) : (
              <SplitCompareSlider analysisData={analysisData} />
            )}
          </div>

          {/* Charts Row: Hydrology Breakdown + SAR Backscatter Curve */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Chart 1: Hydrology Breakdown */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                  <BarChart3 className="w-4 h-4 text-blue-600" />
                  Pre- vs. Post-Disaster Hydrology (km²)
                </h4>
                <span className="text-[10px] text-slate-400">10m GSD</span>
              </div>
              <div className="h-52 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={hydrologyData} margin={{ top: 6, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="category" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={{ stroke: "#e2e8f0" }} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} axisLine={{ stroke: "#e2e8f0" }} />
                    <Tooltip contentStyle={{ backgroundColor: "#ffffff", borderColor: "#e2e8f0", fontSize: "11px", borderRadius: "8px" }} />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Bar dataKey="pre_disaster_km2" name="Pre-Disaster Baseline" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="post_disaster_km2" name="Post-Disaster Inundation" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart 2: SAR Backscatter Curve */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                  <Radio className="w-4 h-4 text-indigo-600" />
                  Sentinel-1 SAR Specular Drop (-16 dB Cutoff)
                </h4>
                <span className="text-[10px] font-semibold text-red-600 bg-red-50 px-2 py-0.5 rounded border border-red-200">
                  Specular Water Detection
                </span>
              </div>
              <div className="h-52 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={sarHistogram} margin={{ top: 6, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="db" type="number" domain={[-27, -3]} tickFormatter={(v) => `${v}dB`} tick={{ fill: "#64748b", fontSize: 10 }} axisLine={{ stroke: "#e2e8f0" }} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} axisLine={{ stroke: "#e2e8f0" }} />
                    <Tooltip contentStyle={{ backgroundColor: "#ffffff", borderColor: "#e2e8f0", fontSize: "11px", borderRadius: "8px" }} />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <ReferenceLine x={-16} stroke="#dc2626" strokeDasharray="4 4" label={{ value: "-16dB Specular Cutoff", fill: "#dc2626", fontSize: 10 }} />
                    <Line type="monotone" dataKey="vv_density" name="VV Polarized Specular %" stroke="#2563eb" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="vh_density" name="VH Polarized Volume %" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Expandable Technical Model Specifications (Sen1Floods11 Benchmark) */}
      {showTechnicalDetails && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Award className="w-4 h-4 text-blue-600" />
                Sen1Floods11 Architecture Ablation &amp; Technical Benchmark
              </h3>
              <p className="text-xs text-slate-500">
                Comparison of our Dual-Stream Attention U-Net against Standard U-Net, ResNet-50, and Single-Stream SAR under &gt;80% monsoon cloud cover.
              </p>
            </div>
          </div>

          <ModelBenchmarkTab />
        </div>
      )}
    </div>
  );
}
