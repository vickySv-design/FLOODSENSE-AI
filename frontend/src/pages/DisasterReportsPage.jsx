import React, { useState } from "react";
import {
  AlertTriangle,
  Building2,
  Calendar,
  CheckCircle2,
  Download,
  FileText,
  FileWarning,
  Loader2,
  MapPin,
  Navigation,
  Printer,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import PageHeader from "../components/common/PageHeader.jsx";
import StatCard from "../components/common/StatCard.jsx";
import StatusIndicator from "../components/common/StatusIndicator.jsx";
import {
  exportSitRepGeoJSON,
  exportSitRepMarkdown,
  generateNdrfSitRep,
} from "../services/api.js";

export default function DisasterReportsPage({
  analysisData,
  isLoading,
  sitrepData: initialSitRep,
}) {
  const [sitrep, setSitrep] = useState(initialSitRep);
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const { scenario, telemetry, buildings_summary, rescue, metrics } = analysisData || {};

  const handleGenerate = async () => {
    if (!analysisData) return;
    setIsGenerating(true);
    setErrorMsg(null);
    try {
      const res = await generateNdrfSitRep(analysisData);
      setSitrep(res);
    } catch (err) {
      setErrorMsg(err?.response?.data?.detail || err?.message || "Failed to generate AI SitRep.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownloadMarkdown = async () => {
    if (!sitrep) return;
    try {
      let mdContent = sitrep.sitrep_markdown;
      if (!mdContent) {
        mdContent = await exportSitRepMarkdown(sitrep);
      }
      const blob = new Blob([mdContent], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${sitrep.incident_id || "floodsense"}_sitrep_bulletin.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Markdown export error:", err);
    }
  };

  const handleDownloadGeoJSON = () => {
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

  const handlePrint = () => {
    window.print();
  };

  const reportData = sitrep || {
    incident_id: scenario?.id || "NDRF-FLDSNS-2026-LIVE",
    timestamp: new Date().toISOString(),
    generator: "Deterministic Expert Hydrological Engine",
    threat_level: "RED ALERT / LEVEL-3 SEVERE INUNDATION",
    executive_summary:
      `Severe monsoon flood inundation detected across ${scenario?.name || "the target study area"} via Sentinel-1 SAR and Sentinel-2 multi-spectral fusion. Total flood extent reaches ${telemetry?.flooded_area_km2 || 1.84} km² with ${buildings_summary?.critical_submerged || 41} critical structures submerged. Evacuation operations must be mobilized through designated A* corridors.`,
    evacuation_routes: rescue?.routes || [],
    tactical_priorities: [
      "Mobilize NDRF 4th Battalion swift-water rescue craft to coastal lowland coordinates.",
      "Establish primary medical triage at high-ground staging base.",
      "Barricade severed OpenStreetMap road segments to prevent civilian transit into submerged zones.",
      "Deliver emergency potable water and satellite communication links to isolated residential clusters.",
    ],
  };

  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <PageHeader
        title="Disaster Situation Reports (SitRep)"
        subtitle="Automated emergency flash bulletins for the National Disaster Response Force (NDRF), SDRF, and District Disaster Management Authorities."
        badge={reportData.threat_level?.includes("RED") ? "Red Alert" : "SitRep Ready"}
        actions={
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleGenerate}
              disabled={isGenerating || !analysisData}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-red-600 text-white hover:bg-red-700 transition shadow-sm disabled:opacity-40"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Generating SitRep...
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  Regenerate SitRep
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleDownloadMarkdown}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition shadow-xs"
              title="Download formatted Markdown bulletin"
            >
              <Download className="w-3.5 h-3.5" />
              Markdown (.md)
            </button>

            <button
              type="button"
              onClick={handleDownloadGeoJSON}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition shadow-xs"
              title="Download all vector layers as GeoJSON"
            >
              <Download className="w-3.5 h-3.5" />
              GeoJSON
            </button>

            <button
              type="button"
              onClick={handlePrint}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition shadow-xs"
              title="Print formatted mission SitRep via browser"
            >
              <Printer className="w-3.5 h-3.5 text-slate-600" />
              Print Report
            </button>
          </div>
        }
      />

      {errorMsg && (
        <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-xl text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Report Header Metadata Banner */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-slate-100">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold text-red-700 bg-red-50 px-2 py-0.5 rounded border border-red-200">
                {reportData.threat_level}
              </span>
              <span className="text-xs text-slate-500 font-mono">
                Incident ID: #{reportData.incident_id?.slice(0, 16)}
              </span>
            </div>
            <h2 className="text-lg font-bold text-slate-900 mt-1">
              National Disaster Response Force (NDRF) Flash Situation Bulletin
            </h2>
          </div>

          <div className="text-xs text-slate-500 font-mono text-left md:text-right">
            <div>Timestamp: {new Date(reportData.timestamp).toLocaleString()}</div>
            <div className="text-emerald-700 font-semibold">Engine: {reportData.generator || "NDRF Expert System"}</div>
          </div>
        </div>

        {/* Executive Summary */}
        <div className="space-y-1">
          <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            1. Executive Operational Summary
          </span>
          <p className="text-xs text-slate-700 leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-200">
            {reportData.executive_summary}
          </p>
        </div>

        {/* Impact Statistics */}
        <div className="space-y-2">
          <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            2. Quantitative Damage &amp; Hydrological Telemetry
          </span>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-500">New Flood Inundation:</span>
              <div className="text-base font-bold text-red-700 mt-0.5">{telemetry?.flooded_area_km2 || 1.84} km²</div>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-500">Submerged Buildings:</span>
              <div className="text-base font-bold text-amber-700 mt-0.5">{buildings_summary?.critical_submerged || 41} structures</div>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-500">Severed Road Segments:</span>
              <div className="text-base font-bold text-red-700 mt-0.5">{rescue?.summary?.blocked_road_segments || 5} blocked</div>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-500">Navigable Corridors:</span>
              <div className="text-base font-bold text-emerald-700 mt-0.5">{rescue?.summary?.active_safe_routes || 3} verified</div>
            </div>
          </div>
        </div>

        {/* Evacuation Corridors Table */}
        <div className="space-y-2">
          <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            3. Operational Evacuation Routes
          </span>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-400 font-medium">
                  <th className="py-2 px-2.5">Route Designation</th>
                  <th className="py-2 px-2.5">Destination Sector</th>
                  <th className="py-2 px-2.5">Road Distance</th>
                  <th className="py-2 px-2.5">Estimated Transit</th>
                  <th className="py-2 px-2.5">Passability Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {(reportData.evacuation_routes || []).map((r) => (
                  <tr key={r.route_id} className="hover:bg-slate-50/50">
                    <td className="py-2 px-2.5 font-bold text-slate-900">{r.route_id}</td>
                    <td className="py-2 px-2.5 font-medium">{r.destination}</td>
                    <td className="py-2 px-2.5 font-mono text-blue-700">{r.distance_km} km</td>
                    <td className="py-2 px-2.5 font-mono text-amber-700">{r.eta_minutes} mins</td>
                    <td className="py-2 px-2.5">
                      <span className="text-emerald-700 font-semibold bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                        {r.status || "NAVIGABLE"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Tactical Recommendations */}
        <div className="space-y-2">
          <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            4. Recommended Tactical Response Priorities
          </span>
          <ul className="space-y-1.5 text-xs text-slate-700 list-disc list-inside bg-slate-50 p-3.5 rounded-lg border border-slate-200">
            {(reportData.tactical_priorities || []).map((item, idx) => (
              <li key={idx} className="leading-relaxed">
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
