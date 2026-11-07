import React from "react";
import {
  Activity,
  BookOpen,
  Cpu,
  Download,
  FileWarning,
  Printer,
  Radar,
  Satellite,
  ShieldAlert,
} from "lucide-react";

export default function HeaderBar({
  health,
  analysisData,
  isLoading,
  onOpenSitRep,
  onExportGeoJSON,
  onPrintSitRep,
  onOpenWalkthrough,
}) {
  const deviceLabel =
    analysisData?.engine?.device || health?.device || "PyTorch Engine";
  const latencyMs = analysisData?.engine?.latency_ms ?? "--";
  const isCuda = Boolean(
    analysisData?.engine?.cuda_available || health?.cuda_available
  );
  const scenarioName =
    analysisData?.scenario?.name || "Initializing Tactical Sensor Feed...";
  const isLiveTelemetry = Boolean(
    analysisData?.scenario?.description?.includes("GEE") ||
    analysisData?.scenario?.gee_telemetry
  );

  return (
    <header className="w-full bg-slate-900/95 border-b border-slate-800 px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 z-30 backdrop-blur-md">
      {/* Left: Project Identity Badge */}
      <div className="flex items-center gap-3">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/40 text-cyan-400">
          <Radar className="w-5 h-5 animate-spin" style={{ animationDuration: "8s" }} />
          <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
          <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-emerald-400" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-bold tracking-wider px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
              FLOODSENSE-AI // DUAL-STREAM SAR-OPTICAL ENGINE
            </span>
            <span className={`hidden md:inline-flex items-center gap-1 text-[11px] font-mono px-2 py-0.5 rounded border ${
              isLiveTelemetry
                ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30"
                : "text-amber-300 bg-amber-500/10 border-amber-500/30"
            }`}>
              <Satellite className="w-3 h-3" />
              {isLiveTelemetry ? "LIVE SATELLITE TELEMETRY" : "CALIBRATED BENCHMARK PRESET"}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5 truncate max-w-md lg:max-w-xl">
            Active AOI:{" "}
            <span className="text-slate-200 font-medium">{scenarioName}</span>
          </p>
        </div>
      </div>

      {/* Center: Hardware Acceleration & Deep Learning Telemetry Badges */}
      <div className="hidden xl:flex items-center gap-2.5">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-950/90 border border-slate-800 text-xs font-mono">
          <Cpu
            className={`w-3.5 h-3.5 ${
              isCuda ? "text-emerald-400" : "text-cyan-400"
            }`}
          />
          <span className="text-slate-400">HW:</span>
          <span className="text-slate-100 font-semibold">{deviceLabel}</span>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-950/90 border border-slate-800 text-xs font-mono">
          <Activity className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-slate-400">LATENCY:</span>
          <span className="text-amber-300 font-semibold">
            {isLoading ? "RUNNING..." : `${latencyMs} ms`}
          </span>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-950/90 border border-slate-800 text-xs font-mono">
          <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
          <span className="text-slate-400">ARCH:</span>
          <span className="text-cyan-300 font-semibold">
            Siamese Attention U-Net + A* Router
          </span>
        </div>
      </div>

      {/* Right: Mission Export & NDRF AI SitRep Actions */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onOpenWalkthrough}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-bold bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 border border-purple-500/50 shadow-[0_0_10px_rgba(168,85,247,0.3)] transition"
          title="Open Step-by-Step Project Review & Defense Guide"
        >
          <BookOpen className="w-3.5 h-3.5 text-purple-400" />
          Viva Guide
        </button>

        <button
          type="button"
          onClick={onOpenSitRep}
          disabled={!analysisData || isLoading}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-bold bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/50 shadow-[0_0_12px_rgba(244,63,94,0.35)] transition disabled:opacity-40 disabled:cursor-not-allowed"
          title="Generate Automated NDRF / District Collector AI Disaster Situation Report"
        >
          <FileWarning className="w-3.5 h-3.5 text-rose-400" />
          Generate NDRF Action SitRep
        </button>

        <button
          type="button"
          onClick={onExportGeoJSON}
          disabled={!analysisData || isLoading}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-semibold bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-300 border border-cyan-500/40 transition disabled:opacity-40 disabled:cursor-not-allowed"
          title="Download all flood polygons, building damage grades, and A* rescue routes as GeoJSON"
        >
          <Download className="w-3.5 h-3.5" />
          Export GeoJSON
        </button>

        <button
          type="button"
          onClick={onPrintSitRep}
          disabled={!analysisData || isLoading}
          className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition disabled:opacity-40 disabled:cursor-not-allowed"
          title="Print formatted Mission Situation Report (SitRep)"
        >
          <Printer className="w-3.5 h-3.5 text-emerald-400" />
          Print Mission SitRep
        </button>
      </div>
    </header>
  );
}
