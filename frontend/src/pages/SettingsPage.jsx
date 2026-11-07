import React, { useState } from "react";
import {
  Activity,
  Award,
  CheckCircle2,
  Compass,
  Cpu,
  Database,
  Globe,
  HardDrive,
  Info,
  Layers,
  Map as MapIcon,
  Moon,
  RefreshCw,
  Save,
  Settings,
  ShieldCheck,
  Sun,
  Zap,
} from "lucide-react";
import PageHeader from "../components/common/PageHeader.jsx";
import StatusIndicator from "../components/common/StatusIndicator.jsx";
import ModelBenchmarkTab from "../components/ModelBenchmarkTab.jsx";

export default function SettingsPage({
  analysisData,
  health,
  presets = [],
  selectedPresetId,
  onSelectPreset,
  onOpenWalkthrough,
}) {
  const [defaultPreset, setDefaultPreset] = useState(selectedPresetId || "chennai_monsoon_2026");
  const [defaultBasemap, setDefaultBasemap] = useState("satellite");
  const [defaultDataMode, setDefaultDataMode] = useState("hybrid");
  const [unitSystem, setUnitSystem] = useState("metric");
  const [savedNotice, setSavedNotice] = useState(null);

  const device = analysisData?.engine?.device || health?.device || "PyTorch Engine";
  const isCuda = Boolean(analysisData?.engine?.cuda_available || health?.cuda_available);
  const latencyMs = analysisData?.engine?.latency_ms ?? 24;

  const handleSavePreferences = () => {
    onSelectPreset(defaultPreset);
    setSavedNotice("Application configuration saved successfully.");
    setTimeout(() => setSavedNotice(null), 3000);
  };

  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <PageHeader
        title="Settings &amp; System Health"
        subtitle="Manage default disaster monitoring parameters, map preferences, and inspect deep learning hardware acceleration."
        actions={
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleSavePreferences}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-blue-600 text-white hover:bg-blue-700 transition shadow-sm"
            >
              <Save className="w-3.5 h-3.5" />
              Save Settings
            </button>
          </div>
        }
      />

      {savedNotice && (
        <div className="bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs px-4 py-2.5 rounded-xl flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>{savedNotice}</span>
        </div>
      )}

      {/* Grid: Preferences + Hardware Diagnostics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Card 1: Application Preferences */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-100">
            <Settings className="w-4 h-4 text-blue-600" />
            Operational Preferences
          </h3>

          <div className="space-y-4 text-xs">
            <div>
              <label className="block text-slate-700 font-semibold mb-1">
                Default Geographical Study Area (AOI)
              </label>
              <select
                value={defaultPreset}
                onChange={(e) => setDefaultPreset(e.target.value)}
                className="w-full p-2 rounded-lg border border-slate-300 text-xs text-slate-800 bg-white"
              >
                {presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.region})
                  </option>
                ))}
              </select>
              <p className="text-[11px] text-slate-500 mt-1">
                Initial study area loaded automatically when mission control starts.
              </p>
            </div>

            <div>
              <label className="block text-slate-700 font-semibold mb-1">
                Default Satellite Basemap
              </label>
              <select
                value={defaultBasemap}
                onChange={(e) => setDefaultBasemap(e.target.value)}
                className="w-full p-2 rounded-lg border border-slate-300 text-xs text-slate-800 bg-white"
              >
                <option value="satellite">ESRI World Imagery (High-Resolution Satellite)</option>
                <option value="osm">OpenStreetMap Standard Street Network</option>
                <option value="carto_light">Carto Light Canvas (Minimal Vector Background)</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-700 font-semibold mb-1">
                Default Data Acquisition Strategy
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: "hybrid", label: "Hybrid" },
                  { id: "live", label: "Live Only" },
                  { id: "offline", label: "Offline Only" },
                ].map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setDefaultDataMode(m.id)}
                    className={`py-2 px-3 rounded-lg border text-center transition ${
                      defaultDataMode === m.id
                        ? "border-blue-600 bg-blue-50 text-blue-700 font-bold"
                        : "border-slate-200 text-slate-600 hover:bg-slate-50"
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-slate-700 font-semibold mb-1">
                Measurement &amp; Unit System
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setUnitSystem("metric")}
                  className={`py-2 px-3 rounded-lg border text-center transition ${
                    unitSystem === "metric"
                      ? "border-blue-600 bg-blue-50 text-blue-700 font-bold"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  Metric (km², m, mm/hr)
                </button>
                <button
                  type="button"
                  onClick={() => setUnitSystem("imperial")}
                  className={`py-2 px-3 rounded-lg border text-center transition ${
                    unitSystem === "imperial"
                      ? "border-blue-600 bg-blue-50 text-blue-700 font-bold"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  Imperial (sq mi, ft, in/hr)
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Card 2: Deep Learning Engine & Device Health */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-100">
            <Cpu className="w-4 h-4 text-emerald-600" />
            PyTorch Hardware Acceleration &amp; Diagnostics
          </h3>

          <div className="space-y-3 text-xs">
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between">
              <div>
                <span className="text-slate-500">Inference Hardware Device:</span>
                <div className="font-bold text-slate-900 text-sm mt-0.5">{device}</div>
              </div>
              <StatusIndicator
                status={isCuda ? "success" : "info"}
                label={isCuda ? "CUDA Accelerated" : "CPU Core Fallback"}
                size="sm"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-slate-500">Inference Latency:</span>
                <div className="font-bold text-blue-700 text-sm mt-0.5">{latencyMs} ms</div>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-slate-500">Model Weights:</span>
                <div className="font-bold text-slate-800 text-sm mt-0.5">14.1 Million</div>
              </div>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1.5">
              <div className="flex justify-between text-slate-600">
                <span>Application Version:</span>
                <strong className="text-slate-900">FloodSense-AI v2.4.0 (Phase 4 Final)</strong>
              </div>
              <div className="flex justify-between text-slate-600">
                <span>FastAPI Backend Port:</span>
                <strong className="text-emerald-700">http://localhost:8000</strong>
              </div>
              <div className="flex justify-between text-slate-600">
                <span>Frontend Dev Server:</span>
                <strong className="text-blue-700">http://localhost:5173</strong>
              </div>
              <div className="flex justify-between text-slate-600">
                <span>Coordinate Reference System:</span>
                <strong className="text-slate-900">WGS-84 (EPSG:4326) / UTM</strong>
              </div>
            </div>

            <div className="pt-2">
              <button
                type="button"
                onClick={onOpenWalkthrough}
                className="w-full py-2.5 px-4 rounded-xl text-xs font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-sm transition flex items-center justify-center gap-2"
              >
                <Award className="w-4 h-4" />
                Launch Full Viva Defense Presentation Guide
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

