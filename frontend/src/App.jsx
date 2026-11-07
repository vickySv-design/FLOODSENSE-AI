import React, { useEffect, useState } from "react";
import { AlertCircle, Columns, Loader2, Map } from "lucide-react";
import HeaderBar from "./components/HeaderBar.jsx";
import ControlSidebar from "./components/ControlSidebar.jsx";
import MapViewport from "./components/MapViewport.jsx";
import SplitCompareSlider from "./components/SplitCompareSlider.jsx";
import AnalyticsDock from "./components/AnalyticsDock.jsx";
import SitRepModal from "./components/SitRepModal.jsx";
import {
  analyzeCustomUpload,
  analyzePresetScenario,
  fetchAndAnalyzeGEETiles,
  fetchDisasterPresets,
  fetchHealthStatus,
  generateNdrfSitRep,
} from "./services/api.js";

export default function App() {
  const [health, setHealth] = useState(null);
  const [presets, setPresets] = useState([]);
  const [selectedPresetId, setSelectedPresetId] = useState(
    "chennai_monsoon_2026"
  );
  const [analysisData, setAnalysisData] = useState(null);
  const [viewportTab, setViewportTab] = useState("map"); // "map" | "split"
  const [isLoading, setIsLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState(null);

  // NDRF AI SitRep Modal State
  const [isSitRepOpen, setIsSitRepOpen] = useState(false);
  const [sitrepData, setSitrepData] = useState(null);
  const [isGeneratingSitRep, setIsGeneratingSitRep] = useState(false);

  const [layerVisibility, setLayerVisibility] = useState({
    allRoads: true,
    newFlood: true,
    permanentWater: true,
    highRiskBuffer: true,
    buildings: true,
    safeRoutes: true,
    severedRoads: true,
  });

  useEffect(() => {
    let mounted = true;
    async function bootMissionControl() {
      setIsLoading(true);
      setErrorMsg(null);
      try {
        const [healthRes, presetsRes] = await Promise.all([
          fetchHealthStatus(),
          fetchDisasterPresets(),
        ]);
        if (!mounted) return;
        setHealth(healthRes);
        setPresets(presetsRes.presets || []);
        const defaultId =
          presetsRes.default_preset_id || "chennai_monsoon_2026";
        setSelectedPresetId(defaultId);

        const initialAnalysis = await analyzePresetScenario(defaultId);
        if (mounted) {
          setAnalysisData(initialAnalysis);
        }
      } catch (err) {
        if (mounted) {
          setErrorMsg(
            err?.response?.data?.detail ||
              err?.message ||
              "Failed to connect to FloodSense-AI Backend at http://localhost:8000"
          );
        }
      } finally {
        if (mounted) {
          setIsLoading(false);
        }
      }
    }
    bootMissionControl();
    return () => {
      mounted = false;
    };
  }, []);

  const handleSelectPreset = async (presetId) => {
    setSelectedPresetId(presetId);
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const result = await analyzePresetScenario(presetId);
      setAnalysisData(result);
    } catch (err) {
      setErrorMsg(
        err?.response?.data?.detail ||
          err?.message ||
          "Error analyzing disaster preset."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadCustomPair = async (uploadPayload) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const result = await analyzeCustomUpload(uploadPayload);
      setSelectedPresetId("custom_upload_mission");
      setAnalysisData(result);
    } catch (err) {
      setErrorMsg(
        err?.response?.data?.detail ||
          err?.message ||
          "Error processing custom uploaded satellite tiles."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleFetchGEE = async (geePayload) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const result = await fetchAndAnalyzeGEETiles(geePayload);
      setSelectedPresetId("gee_live_acquisition");
      setAnalysisData(result);
    } catch (err) {
      setErrorMsg(
        err?.response?.data?.detail ||
          err?.message ||
          "Error fetching Copernicus Sentinel-1/2 tiles."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleOpenSitRep = async () => {
    if (!analysisData) return;
    setIsSitRepOpen(true);
    setIsGeneratingSitRep(true);
    try {
      const res = await generateNdrfSitRep(analysisData);
      setSitrepData(res);
    } catch (err) {
      setErrorMsg(
        err?.response?.data?.detail ||
          err?.message ||
          "Failed to generate NDRF Situation Report."
      );
    } finally {
      setIsGeneratingSitRep(false);
    }
  };

  const handleToggleLayer = (layerKey) => {
    setLayerVisibility((prev) => ({
      ...prev,
      [layerKey]: !prev[layerKey],
    }));
  };

  const handleExportGeoJSON = () => {
    if (!analysisData?.export_geojson) return;
    const blob = new Blob(
      [JSON.stringify(analysisData.export_geojson, null, 2)],
      { type: "application/geo+json" }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const safeId = analysisData?.scenario?.id || "floodsense_mission";
    a.href = url;
    a.download = `${safeId}_sitrep_vectors.geojson`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handlePrintSitRep = () => {
    window.print();
  };

  return (
    <div className="min-h-screen lg:h-screen w-full bg-slate-950 text-slate-100 flex flex-col overflow-x-hidden lg:overflow-hidden">
      {/* Top Telemetry Navbar */}
      <HeaderBar
        health={health}
        analysisData={analysisData}
        isLoading={isLoading}
        onOpenSitRep={handleOpenSitRep}
        onExportGeoJSON={handleExportGeoJSON}
        onPrintSitRep={handlePrintSitRep}
      />

      {/* Main Tactical Command Workspace */}
      <div className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-hidden">
        {/* Left Tactical Control Sidebar */}
        <ControlSidebar
          presets={presets}
          selectedPresetId={selectedPresetId}
          onSelectPreset={handleSelectPreset}
          onUploadCustomPair={handleUploadCustomPair}
          onFetchGEE={handleFetchGEE}
          layerVisibility={layerVisibility}
          onToggleLayer={handleToggleLayer}
          isLoading={isLoading}
        />

        {/* Center & Bottom Multi-Modal Viewport + Analytics Dock */}
        <main className="flex-1 flex flex-col min-w-0 min-h-0 overflow-y-auto lg:overflow-hidden">
          {/* Error Banner if any */}
          {errorMsg && (
            <div className="bg-rose-950/90 border-b border-rose-500/50 px-4 py-2 text-xs font-mono text-rose-200 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Viewport Mode Switcher Bar */}
          <div className="bg-slate-900/90 border-b border-slate-800 px-4 py-2 flex flex-wrap items-center justify-between gap-2 shrink-0">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => setViewportTab("map")}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-semibold transition ${
                  viewportTab === "map"
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                    : "text-slate-400 hover:text-slate-200 bg-slate-950 border border-slate-800"
                }`}
              >
                <Map className="w-3.5 h-3.5" />
                View 1: Live Tactical GIS Map (Leaflet + A* Router)
              </button>

              <button
                type="button"
                onClick={() => setViewportTab("split")}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-semibold transition ${
                  viewportTab === "split"
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                    : "text-slate-400 hover:text-slate-200 bg-slate-950 border border-slate-800"
                }`}
              >
                <Columns className="w-3.5 h-3.5" />
                View 2: Split-Screen Before/After SAR-Optical Curtain
              </button>
            </div>

            {analysisData?.scenario && (
              <div className="text-[11px] font-mono text-slate-400 flex items-center gap-3">
                <span>
                  Sensor:{" "}
                  <strong className="text-slate-200">
                    {analysisData.scenario.sensor_pair}
                  </strong>
                </span>
                <span className="hidden md:inline text-slate-700">|</span>
                <span className="hidden md:inline">
                  Pre: {analysisData.scenario.pre_date} → Post:{" "}
                  {analysisData.scenario.post_date}
                </span>
              </div>
            )}
          </div>

          {/* Center Interactive Viewport */}
          <div className="relative flex-1 min-h-[380px]">
            {isLoading && (
              <div className="absolute inset-0 z-30 bg-slate-950/75 backdrop-blur-sm flex flex-col items-center justify-center gap-2">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
                <div className="text-xs font-mono text-cyan-300 font-semibold">
                  EXECUTING DUAL-STREAM ATTENTION U-NET & A* EVACUATION ROUTER...
                </div>
              </div>
            )}

            {viewportTab === "map" ? (
              <MapViewport
                analysisData={analysisData}
                layerVisibility={layerVisibility}
              />
            ) : (
              <SplitCompareSlider analysisData={analysisData} />
            )}
          </div>

          {/* Bottom Analytics & Telemetry Dock */}
          <AnalyticsDock
            analysisData={analysisData}
            selectedPresetId={selectedPresetId}
            onOpenSitRep={handleOpenSitRep}
          />
        </main>
      </div>

      {/* Automated NDRF / District Collector AI SitRep Modal */}
      <SitRepModal
        isOpen={isSitRepOpen}
        onClose={() => setIsSitRepOpen(false)}
        sitrepData={sitrepData}
        isGenerating={isGeneratingSitRep}
      />
    </div>
  );
}
