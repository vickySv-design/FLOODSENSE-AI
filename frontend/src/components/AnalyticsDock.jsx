import React, { useState } from "react";
import {
  AlertTriangle,
  Award,
  BarChart3,
  Building2,
  CheckCircle2,
  Clock,
  Droplets,
  FileWarning,
  Navigation,
  Radio,
  Users,
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
import TimelineSimulatorPanel from "./TimelineSimulatorPanel.jsx";
import ModelBenchmarkTab from "./ModelBenchmarkTab.jsx";

export default function AnalyticsDock({
  analysisData,
  selectedPresetId,
  onOpenSitRep,
}) {
  const [activeDockTab, setActiveDockTab] = useState("telemetry"); // "telemetry" | "timeline" | "benchmarks"

  if (!analysisData) {
    return null;
  }

  const { telemetry, buildings_summary, rescue, metrics, charts } = analysisData;
  const hydrologyData = charts?.hydrology_breakdown || [];
  const sarHistogram = charts?.sar_backscatter_curve || [];

  return (
    <section className="w-full bg-slate-900/95 border-t border-slate-800 p-3 space-y-3">
      {/* Row 1: 4 High-Impact Tactical KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
        {/* KPI 1: New Flood Inundation */}
        <div className="p-3 rounded-lg bg-slate-950/90 border border-rose-500/30 relative overflow-hidden">
          <div className="flex items-start justify-between">
            <div>
              <span className="text-[10px] font-mono uppercase tracking-wider text-rose-400 font-semibold">
                New Flood Inundation (Class 2)
              </span>
              <div className="text-2xl font-mono font-bold text-slate-100 mt-1">
                {telemetry?.flooded_area_km2}{" "}
                <span className="text-sm font-normal text-slate-400">km²</span>
              </div>
            </div>
            <div className="p-2 rounded-lg bg-rose-500/15 text-rose-400 border border-rose-500/30">
              <Droplets className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono">
            <span className="text-rose-300 font-semibold">
              +{telemetry?.flood_expansion_pct}% vs. Baseline Water
            </span>
            <span className="text-slate-400">
              Perm: {telemetry?.permanent_water_km2} km²
            </span>
          </div>
        </div>

        {/* KPI 2: Submerged / Critical Buildings */}
        <div className="p-3 rounded-lg bg-slate-950/90 border border-amber-500/30 relative overflow-hidden">
          <div className="flex items-start justify-between">
            <div>
              <span className="text-[10px] font-mono uppercase tracking-wider text-amber-400 font-semibold">
                Submerged / Critical Structures
              </span>
              <div className="text-2xl font-mono font-bold text-slate-100 mt-1">
                {buildings_summary?.critical_submerged}{" "}
                <span className="text-xs font-normal text-amber-300">
                  + {buildings_summary?.moderate_buffer_risk} Buffer Risk
                </span>
              </div>
            </div>
            <div className="p-2 rounded-lg bg-amber-500/15 text-amber-400 border border-amber-500/30">
              <Building2 className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono">
            <span className="text-slate-200 flex items-center gap-1">
              <Users className="w-3 h-3 text-cyan-400" />
              Est. Affected:{" "}
              <strong className="text-amber-300">
                {buildings_summary?.estimated_affected_population?.toLocaleString()}
              </strong>
            </span>
            <span className="text-emerald-400">
              {buildings_summary?.safe_dry} Safe
            </span>
          </div>
        </div>

        {/* KPI 3: Dynamic A* Rescue Routing Status */}
        <div className="p-3 rounded-lg bg-slate-950/90 border border-emerald-500/30 relative overflow-hidden">
          <div className="flex items-start justify-between">
            <div>
              <span className="text-[10px] font-mono uppercase tracking-wider text-emerald-400 font-semibold">
                A* Rescue Routing Telemetry
              </span>
              <div className="text-2xl font-mono font-bold text-slate-100 mt-1">
                {rescue?.summary?.active_safe_routes} Safe Corridors{" "}
                <span className="text-xs font-normal text-emerald-300">
                  ({rescue?.summary?.shortest_route_km} km min)
                </span>
              </div>
            </div>
            <div className="p-2 rounded-lg bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
              <Navigation className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono">
            <span className="text-emerald-300">
              Fastest ETA: {rescue?.summary?.fastest_eta_mins} mins
            </span>
            <span className="text-rose-400 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" />
              {rescue?.summary?.blocked_road_segments} Severed Roads
            </span>
          </div>
        </div>

        {/* KPI 4: Deep Learning Segmentation Metrics */}
        <div className="p-3 rounded-lg bg-slate-950/90 border border-cyan-500/30 relative overflow-hidden">
          <div className="flex items-start justify-between">
            <div>
              <span className="text-[10px] font-mono uppercase tracking-wider text-cyan-400 font-semibold">
                Siamese Attention U-Net Metrics
              </span>
              <div className="text-2xl font-mono font-bold text-slate-100 mt-1">
                {((metrics?.miou || 0) * 100).toFixed(1)}%{" "}
                <span className="text-xs font-normal text-cyan-300">mIoU</span>
                <span className="text-slate-600 mx-1.5">|</span>
                {((metrics?.f1_score || 0) * 100).toFixed(1)}%{" "}
                <span className="text-xs font-normal text-emerald-300">
                  Dice/F1
                </span>
              </div>
            </div>
            <div className="p-2 rounded-lg bg-cyan-500/15 text-cyan-400 border border-cyan-500/30">
              <CheckCircle2 className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono">
            <span className="text-slate-300">
              Precision:{" "}
              <strong className="text-cyan-300">
                {((metrics?.precision || 0) * 100).toFixed(1)}%
              </strong>
            </span>
            <span className="text-slate-300">
              Recall:{" "}
              <strong className="text-emerald-300">
                {((metrics?.recall || 0) * 100).toFixed(1)}%
              </strong>
            </span>
          </div>
        </div>
      </div>

      {/* Analytics Dock Navigation Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => setActiveDockTab("telemetry")}
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded text-xs font-mono font-semibold transition ${
              activeDockTab === "telemetry"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200"
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            Hydrology & SAR Telemetry
          </button>

          <button
            type="button"
            onClick={() => setActiveDockTab("timeline")}
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded text-xs font-mono font-semibold transition ${
              activeDockTab === "timeline"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200"
            }`}
          >
            <Clock className="w-3.5 h-3.5 text-amber-400" />
            48-Hour Flood Progression Simulator
          </button>

          <button
            type="button"
            onClick={() => setActiveDockTab("benchmarks")}
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded text-xs font-mono font-semibold transition ${
              activeDockTab === "benchmarks"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200"
            }`}
          >
            <Award className="w-3.5 h-3.5 text-emerald-400" />
            Architecture Comparison (Viva Defense)
          </button>
        </div>

        <button
          type="button"
          onClick={onOpenSitRep}
          className="inline-flex items-center gap-1.5 px-3 py-1 rounded text-xs font-mono font-bold bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/50 transition"
        >
          <FileWarning className="w-3.5 h-3.5 text-rose-400" />
          Generate NDRF Action SitRep
        </button>
      </div>

      {/* Tab 1: Live Recharts Analytical Visualizations + Active Evacuation Route Table */}
      {activeDockTab === "telemetry" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
          {/* Chart 1: Pre-Disaster vs. Post-Disaster Hydrology Breakdown (km²) */}
          <div className="p-3 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-semibold text-slate-200 flex items-center gap-1.5">
                <BarChart3 className="w-3.5 h-3.5 text-cyan-400" />
                Pre- vs. Post-Disaster Hydrology Breakdown (km²)
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                10m/px Area
              </span>
            </div>
            <div className="h-44 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={hydrologyData}
                  margin={{ top: 6, right: 10, left: -18, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="category"
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    axisLine={{ stroke: "#334155" }}
                  />
                  <YAxis
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    axisLine={{ stroke: "#334155" }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      fontSize: "11px",
                      borderRadius: "6px",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px" }} />
                  <Bar
                    dataKey="pre_disaster_km2"
                    name="Pre-Disaster (km²)"
                    fill="#06b6d4"
                    radius={[3, 3, 0, 0]}
                  />
                  <Bar
                    dataKey="post_disaster_km2"
                    name="Post-Disaster (km²)"
                    fill="#f43f5e"
                    radius={[3, 3, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Chart 2: Sentinel-1 SAR VV/VH Backscatter Histogram (dB) */}
          <div className="p-3 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-semibold text-slate-200 flex items-center gap-1.5">
                <Radio className="w-3.5 h-3.5 text-rose-400" />
                Sentinel-1 SAR VV/VH Backscatter Curve (dB)
              </span>
              <span className="text-[10px] font-mono text-rose-400 bg-rose-500/10 px-1.5 py-0.5 rounded">
                -16 dB Water Cutoff
              </span>
            </div>
            <div className="h-44 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={sarHistogram}
                  margin={{ top: 6, right: 12, left: -18, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="db"
                    type="number"
                    domain={[-27, -3]}
                    tickFormatter={(v) => `${v}dB`}
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    axisLine={{ stroke: "#334155" }}
                  />
                  <YAxis
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    axisLine={{ stroke: "#334155" }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      fontSize: "11px",
                      borderRadius: "6px",
                    }}
                    labelFormatter={(v) => `Backscatter: ${v} dB`}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px" }} />
                  <ReferenceLine
                    x={-16}
                    stroke="#f43f5e"
                    strokeDasharray="4 4"
                    label={{
                      value: "-16dB Specular Water",
                      fill: "#fb7185",
                      fontSize: 10,
                      position: "insideTopLeft",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="vv_density"
                    name="SAR VV Polarization (%)"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="vh_density"
                    name="SAR VH Polarization (%)"
                    stroke="#a855f7"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Panel 3: Dynamic A* Evacuation Corridor Dispatch Manifest */}
          <div className="p-3 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-semibold text-slate-200 flex items-center gap-1.5">
                  <Navigation className="w-3.5 h-3.5 text-emerald-400" />
                  A* Evacuation Corridor Manifest
                </span>
                <span className="text-[10px] font-mono text-emerald-400">
                  Road Integrity: {rescue?.summary?.network_integrity_pct}%
                </span>
              </div>

              <div className="space-y-2">
                {rescue?.routes?.map((rt) => (
                  <div
                    key={rt.route_id}
                    className="p-2 rounded bg-slate-900/90 border border-slate-800 flex items-center justify-between gap-2 text-xs"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ backgroundColor: rt.color }}
                        />
                        <span className="font-mono font-bold text-slate-100">
                          {rt.route_id}
                        </span>
                        <span className="text-slate-300 truncate">
                          {rt.destination}
                        </span>
                      </div>
                      <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                        Status: {rt.status} ({rt.waypoints_count} nodes)
                      </div>
                    </div>
                    <div className="text-right font-mono shrink-0">
                      <div className="text-cyan-300 font-bold">
                        {rt.distance_km} km
                      </div>
                      <div className="text-[10px] text-amber-300">
                        ETA {rt.eta_minutes}m
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-400">
              <span>Origin: NDRF Tactical Staging Base</span>
              <span className="text-cyan-400">Heuristic: Haversine A*</span>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: 48-Hour Flood Recession & Surge Forecast Simulator */}
      {activeDockTab === "timeline" && (
        <TimelineSimulatorPanel presetId={selectedPresetId} />
      )}

      {/* Tab 3: Academic Model Ablation & Benchmarking Suite */}
      {activeDockTab === "benchmarks" && <ModelBenchmarkTab />}
    </section>
  );
}
