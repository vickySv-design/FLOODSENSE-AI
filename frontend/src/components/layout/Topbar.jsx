import React from "react";
import {
  Activity,
  Award,
  BookOpen,
  Compass,
  Cpu,
  FileWarning,
  Menu,
  Satellite,
  ShieldCheck,
} from "lucide-react";
import StatusIndicator from "../common/StatusIndicator.jsx";

export default function Topbar({
  activePage,
  pageTitle,
  analysisData,
  health,
  isLoading,
  onOpenMobileMenu,
  onOpenAOIModal,
  onOpenSitRep,
  onOpenWalkthrough,
}) {
  const scenarioName = analysisData?.scenario?.name || "Global AOI Workspace";
  const bbox = analysisData?.scenario?.bbox;
  const device = analysisData?.engine?.device || health?.device || "PyTorch Engine";
  const isCuda = Boolean(analysisData?.engine?.cuda_available || health?.cuda_available);
  const latencyMs = analysisData?.engine?.latency_ms ?? (isLoading ? "..." : 24);

  const isLive = Boolean(
    analysisData?.scenario?.description?.includes("GEE") ||
    analysisData?.scenario?.description?.includes("Dynamic") ||
    analysisData?.scenario?.gee_telemetry
  );

  return (
    <header className="h-16 bg-white border-b border-slate-200 px-4 lg:px-6 flex items-center justify-between gap-4 sticky top-0 z-30 shadow-xs">
      {/* Left: Mobile Toggle & Page Info */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onOpenMobileMenu}
          className="lg:hidden p-2 rounded-lg text-slate-600 hover:bg-slate-100 transition"
          title="Open Menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="hidden sm:block">
          <div className="text-xs text-slate-400 font-medium">FloodSense-AI Operations</div>
          <div className="text-base font-bold text-slate-900 leading-tight">
            {pageTitle}
          </div>
        </div>
      </div>

      {/* Center: Active AOI Study Area Pill */}
      <div className="flex items-center gap-2">
        <div className="bg-slate-50 border border-slate-200 rounded-xl px-3 py-1.5 flex items-center gap-2.5 max-w-xs md:max-w-md">
          <Compass className="w-4 h-4 text-blue-600 shrink-0" />
          <div className="min-w-0">
            <div className="text-[10px] text-slate-500 font-medium flex items-center gap-1.5 truncate">
              <span>Study Area (AOI):</span>
              <span className="font-semibold text-slate-800 truncate">{scenarioName}</span>
            </div>
            {bbox && (
              <div className="text-[10px] text-slate-400 font-mono truncate hidden md:block">
                [{bbox[0].toFixed(2)}°, {bbox[1].toFixed(2)}° to {bbox[2].toFixed(2)}°, {bbox[3].toFixed(2)}°]
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={onOpenAOIModal}
            className="text-xs font-semibold text-blue-600 hover:text-blue-800 hover:underline shrink-0 ml-1"
            title="Configure or Switch Geographical Study Area"
          >
            Change
          </button>
        </div>
      </div>

      {/* Right: Telemetry & Actions */}
      <div className="flex items-center gap-2.5">
        {/* Hardware Status Pill */}
        <div className="hidden xl:flex items-center gap-2 px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200 text-xs">
          <Cpu className={`w-3.5 h-3.5 ${isCuda ? "text-emerald-600" : "text-blue-600"}`} />
          <span className="text-slate-500">Device:</span>
          <span className="font-semibold text-slate-800">{device}</span>
          <span className="text-slate-300">|</span>
          <span className="text-slate-500">{latencyMs} ms</span>
        </div>

        {/* Live / Benchmark Status Indicator */}
        <div className="hidden md:block">
          <StatusIndicator
            status={isLive ? "success" : "info"}
            label={isLive ? "Live Telemetry" : "Calibrated Preset"}
            size="sm"
          />
        </div>

        {/* Quick NDRF SitRep Action */}
        <button
          type="button"
          onClick={onOpenSitRep}
          disabled={!analysisData || isLoading}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-50 text-red-700 border border-red-200 hover:bg-red-100 transition disabled:opacity-40 disabled:cursor-not-allowed shadow-xs"
          title="Generate Automated NDRF Action SitRep"
        >
          <FileWarning className="w-3.5 h-3.5 text-red-600" />
          <span className="hidden sm:inline">NDRF SitRep</span>
        </button>

        {/* Viva Guide Launcher */}
        <button
          type="button"
          onClick={onOpenWalkthrough}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 text-white hover:bg-blue-700 transition shadow-sm"
          title="Review Guided Viva Presentation"
        >
          <Award className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Viva Guide</span>
        </button>
      </div>
    </header>
  );
}

