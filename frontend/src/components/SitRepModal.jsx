import React from "react";
import {
  AlertOctagon,
  Bus,
  CheckCircle2,
  Download,
  FileText,
  HeartPulse,
  LifeBuoy,
  PackageCheck,
  Printer,
  ShieldAlert,
  Sparkles,
  X,
} from "lucide-react";

export default function SitRepModal({
  isOpen,
  onClose,
  sitrepData,
  isGenerating,
}) {
  if (!isOpen) return null;

  const severity = sitrepData?.severity_level || "CRITICAL (RED ALERT)";
  const isRed = severity.includes("CRITICAL");
  const isOrange = severity.includes("HIGH");

  const badgeStyle = isRed
    ? "bg-rose-500/20 text-rose-300 border-rose-500/50 shadow-[0_0_15px_rgba(244,63,94,0.35)]"
    : isOrange
    ? "bg-amber-500/20 text-amber-300 border-amber-500/50"
    : "bg-yellow-500/20 text-yellow-300 border-yellow-500/50";

  const handleDownloadMarkdown = () => {
    if (!sitrepData?.markdown_report) return;
    const blob = new Blob([sitrepData.markdown_report], {
      type: "text/markdown;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${sitrepData.bulletin_id || "NDRF_SITREP"}_Briefing.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handlePrintBriefing = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4 overflow-y-auto">
      <div className="w-full max-w-4xl bg-slate-900 border border-slate-700/90 rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Top Modal Header */}
        <div className="px-5 py-3.5 bg-slate-950 border-b border-slate-800 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-rose-500/15 border border-rose-500/40 text-rose-400">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-mono font-bold text-slate-100">
                  NDRF / SDMA AUTOMATED DISASTER SITUATION REPORT (SITREP)
                </h2>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                  {sitrepData?.bulletin_id || "NDRF-FLASH-2026"}
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Engine: {sitrepData?.generation_engine || "NDRF Expert System"} |{" "}
                {sitrepData?.timestamp_utc}
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

        {/* Body Content */}
        <div className="p-5 overflow-y-auto space-y-5">
          {isGenerating ? (
            <div className="py-16 text-center space-y-3">
              <Sparkles className="w-8 h-8 text-cyan-400 animate-spin mx-auto" />
              <div className="text-xs font-mono text-cyan-300">
                SYNTHESIZING NDRF FLASH BULLETIN & LOGISTICS ALLOCATION...
              </div>
            </div>
          ) : (
            <>
              {/* Alert Banner + AOI Header */}
              <div className="p-3.5 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-xs font-mono text-slate-400">
                    TARGET DISASTER SECTOR
                  </div>
                  <div className="text-sm font-bold text-slate-100 mt-0.5">
                    {sitrepData?.scenario_name} —{" "}
                    <span className="text-cyan-300">{sitrepData?.region}</span>
                  </div>
                </div>
                <div
                  className={`px-3 py-1.5 rounded-md border font-mono text-xs font-bold flex items-center gap-1.5 ${badgeStyle}`}
                >
                  <AlertOctagon className="w-4 h-4" />
                  {severity}
                </div>
              </div>

              {/* Section 1: Executive Hydrological Summary */}
              <div className="p-4 rounded-lg bg-slate-950/70 border border-cyan-500/30 space-y-1.5">
                <div className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-bold flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5" />
                  1. Executive Multi-Sensor Hydrological Assessment
                </div>
                <p className="text-xs text-slate-200 leading-relaxed">
                  {sitrepData?.executive_summary}
                </p>
              </div>

              {/* Section 2: Calculated Resource Allocation Cards */}
              <div>
                <div className="text-xs font-mono uppercase tracking-wider text-emerald-400 font-bold mb-2.5">
                  2. Calculated Emergency Logistics & Resource Manifest (24h Window)
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <div className="flex items-center justify-between text-cyan-400 mb-1">
                      <span className="text-[10px] font-mono uppercase">
                        Inflatable Boats
                      </span>
                      <LifeBuoy className="w-4 h-4" />
                    </div>
                    <div className="text-xl font-mono font-bold text-slate-100">
                      {sitrepData?.resource_allocation?.inflatable_boats_needed}{" "}
                      <span className="text-xs font-normal text-slate-400">
                        Units
                      </span>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <div className="flex items-center justify-between text-rose-400 mb-1">
                      <span className="text-[10px] font-mono uppercase">
                        Medical Teams
                      </span>
                      <HeartPulse className="w-4 h-4" />
                    </div>
                    <div className="text-xl font-mono font-bold text-slate-100">
                      {sitrepData?.resource_allocation?.medical_teams}{" "}
                      <span className="text-xs font-normal text-slate-400">
                        Teams
                      </span>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <div className="flex items-center justify-between text-amber-400 mb-1">
                      <span className="text-[10px] font-mono uppercase">
                        24h Food Packets
                      </span>
                      <PackageCheck className="w-4 h-4" />
                    </div>
                    <div className="text-xl font-mono font-bold text-slate-100">
                      {sitrepData?.resource_allocation?.food_packets_24h?.toLocaleString()}{" "}
                      <span className="text-xs font-normal text-slate-400">
                        Kits
                      </span>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <div className="flex items-center justify-between text-emerald-400 mb-1">
                      <span className="text-[10px] font-mono uppercase">
                        Evacuation Buses
                      </span>
                      <Bus className="w-4 h-4" />
                    </div>
                    <div className="text-xl font-mono font-bold text-slate-100">
                      {sitrepData?.resource_allocation?.evacuation_buses}{" "}
                      <span className="text-xs font-normal text-slate-400">
                        Buses
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Section 3: NDRF Tactical Deployment Orders */}
              <div>
                <div className="text-xs font-mono uppercase tracking-wider text-amber-400 font-bold mb-2.5">
                  3. District Collector & NDRF Operational Directives
                </div>
                <div className="space-y-2">
                  {sitrepData?.ndrf_deployment_orders?.map((order, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-lg bg-slate-950/90 border border-slate-800 flex items-start gap-2.5"
                    >
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      <p className="text-xs text-slate-200 font-mono leading-relaxed">
                        {order}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-5 py-3 bg-slate-950 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <span className="text-[11px] font-mono text-slate-400">
            Authorized for NDRF / SDMA Emergency Command Dispatch
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleDownloadMarkdown}
              disabled={isGenerating || !sitrepData}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-md text-xs font-mono font-semibold bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 transition"
            >
              <Download className="w-3.5 h-3.5" />
              Download Official Markdown Briefing
            </button>
            <button
              type="button"
              onClick={handlePrintBriefing}
              disabled={isGenerating || !sitrepData}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-md text-xs font-mono font-semibold bg-emerald-500 hover:bg-emerald-400 text-slate-950 transition"
            >
              <Printer className="w-3.5 h-3.5" />
              Print Official PDF Briefing
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

