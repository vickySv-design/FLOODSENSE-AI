import React, { useEffect, useState } from "react";
import {
  Clock,
  CloudRain,
  Loader2,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { simulateFloodTimeline } from "../services/api.js";

export default function TimelineSimulatorPanel({
  presetId,
  onForecastUpdate,
}) {
  const [rainfallMmHr, setRainfallMmHr] = useState(65);
  const [horizonHours, setHorizonHours] = useState(24);
  const [simData, setSimData] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);

  const validPresetId =
    presetId &&
    [
      "chennai_monsoon_2026",
      "assam_brahmaputra_surge",
      "wayanad_kerala_flashflood",
    ].includes(presetId)
      ? presetId
      : "chennai_monsoon_2026";

  useEffect(() => {
    let active = true;
    const timer = setTimeout(async () => {
      setIsSimulating(true);
      try {
        const res = await simulateFloodTimeline({
          preset_id: validPresetId,
          rainfall_mm_hr: rainfallMmHr,
          horizon_hours: horizonHours,
        });
        if (active) {
          setSimData(res);
          if (onForecastUpdate) {
            onForecastUpdate(res);
          }
        }
      } catch (err) {
        console.error("Timeline simulation error:", err);
      } finally {
        if (active) setIsSimulating(false);
      }
    }, 180);

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [validPresetId, rainfallMmHr, horizonHours]);

  const horizons = [0, 6, 12, 24, 48];
  const selectedProj = simData?.selected_projection;
  const deltaPct = selectedProj?.delta_from_t0_pct || 0;
  const isSurge = deltaPct >= 0;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
      {/* Left Column: Controls (Rainfall Intensity Slider + Horizon Selector) */}
      <div className="lg:col-span-4 p-3 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col justify-between space-y-3">
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono font-bold text-cyan-300 flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-cyan-400" />
              48-Hour Hydrological Forecaster
            </span>
            {isSimulating && (
              <Loader2 className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
            )}
          </div>

          {/* Rainfall Intensity Slider */}
          <div className="p-2.5 rounded-md bg-slate-900/90 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-300 flex items-center gap-1">
                <CloudRain className="w-3.5 h-3.5 text-cyan-400" />
                Rainfall Intensity:
              </span>
              <span className="text-cyan-300 font-bold">
                {rainfallMmHr} mm/hr
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="150"
              step="5"
              value={rainfallMmHr}
              onChange={(e) => setRainfallMmHr(Number(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <div className="flex flex-wrap gap-1 pt-0.5">
              {[
                { label: "0mm (Drainage)", val: 0 },
                { label: "20mm (Steady)", val: 20 },
                { label: "75mm (Monsoon)", val: 75 },
                { label: "135mm (Cloudburst)", val: 135 },
              ].map((btn) => (
                <button
                  key={btn.val}
                  type="button"
                  onClick={() => setRainfallMmHr(btn.val)}
                  className={`px-2 py-0.5 rounded text-[10px] font-mono border transition ${
                    rainfallMmHr === btn.val
                      ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40"
                      : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                  }`}
                >
                  {btn.label}
                </button>
              ))}
            </div>
          </div>

          {/* Horizon Selector Buttons */}
          <div className="mt-2.5">
            <div className="text-[10px] font-mono uppercase text-slate-400 mb-1.5">
              Forecast Horizon Selector
            </div>
            <div className="grid grid-cols-5 gap-1">
              {horizons.map((hr) => (
                <button
                  key={hr}
                  type="button"
                  onClick={() => setHorizonHours(hr)}
                  className={`py-1.5 rounded text-xs font-mono font-semibold border transition ${
                    horizonHours === hr
                      ? "bg-rose-500/20 text-rose-300 border-rose-500/50"
                      : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
                  }`}
                >
                  {hr === 0 ? "T+0h" : `+${hr}h`}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Selected Horizon Summary Pill */}
        {selectedProj && (
          <div className="p-2.5 rounded-md bg-slate-900 border border-slate-800 flex items-center justify-between text-xs font-mono">
            <div>
              <div className="text-[10px] text-slate-400">
                PROJECTED @ T+{horizonHours}h
              </div>
              <div className="text-sm font-bold text-slate-100">
                {selectedProj.projected_flood_km2} km²{" "}
                <span
                  className={
                    isSurge ? "text-rose-400 text-xs" : "text-emerald-400 text-xs"
                  }
                >
                  ({isSurge ? `+${deltaPct}%` : `${deltaPct}%`})
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] text-slate-400">AT-RISK BLDGS</div>
              <div className="text-sm font-bold text-amber-300 flex items-center justify-end gap-1">
                {isSurge ? (
                  <TrendingUp className="w-3.5 h-3.5 text-rose-400" />
                ) : (
                  <TrendingDown className="w-3.5 h-3.5 text-emerald-400" />
                )}
                {selectedProj.at_risk_buildings} Structures
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Center Column: 48-Hour Flood Progression Curve */}
      <div className="lg:col-span-5 p-3 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-mono font-semibold text-slate-200">
            Spatio-Temporal Flood Trajectory (T+0h to T+48h)
          </span>
          <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded">
            {simData?.hydrological_regime?.split(" ")[0] || "DEM PROPAGATION"}
          </span>
        </div>
        <div className="h-44 w-full flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={simData?.timeline || []}
              margin={{ top: 6, right: 12, left: -16, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="horizon_label"
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
              <Area
                type="monotone"
                dataKey="projected_flood_km2"
                name="Projected Flood Area (km²)"
                stroke="#f43f5e"
                fill="#f43f5e"
                fillOpacity={0.25}
                strokeWidth={2}
              />
              <Area
                type="monotone"
                dataKey="buffer_risk_km2"
                name="100m High-Risk Buffer (km²)"
                stroke="#f59e0b"
                fill="#f59e0b"
                fillOpacity={0.18}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Right Column: Live DEM-Propagated Forecast Mask Preview */}
      <div className="lg:col-span-3 p-3 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col items-center justify-between">
        <div className="w-full flex items-center justify-between text-xs font-mono mb-1.5">
          <span className="text-slate-200 font-semibold">
            Forecast Mask (T+{horizonHours}h)
          </span>
          <span className="text-[10px] text-rose-400">DEM-Weighted</span>
        </div>
        {simData?.forecast_overlay_b64 ? (
          <img
            src={simData.forecast_overlay_b64}
            alt={`Projected Flood Mask T+${horizonHours}h`}
            className="w-36 h-36 rounded-lg object-cover border border-slate-700 shadow-md"
          />
        ) : (
          <div className="w-36 h-36 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center text-[10px] font-mono text-slate-500">
            Simulating...
          </div>
        )}
        <div className="w-full text-center text-[10px] font-mono text-slate-400 mt-1.5">
          Submerged Bldgs:{" "}
          <strong className="text-rose-400">
            {selectedProj?.critical_submerged_buildings ?? "--"}
          </strong>{" "}
          | Buffer:{" "}
          <strong className="text-amber-300">
            {selectedProj?.buffer_risk_buildings ?? "--"}
          </strong>
        </div>
      </div>
    </div>
  );
}

