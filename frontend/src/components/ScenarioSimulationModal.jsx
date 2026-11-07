import React, { useState, useEffect } from "react";
import {
  Wind,
  CloudRain,
  Compass,
  Play,
  RotateCcw,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  X,
  Loader2,
  Download,
  FileText,
  Activity,
  Layers,
  ShieldAlert,
} from "lucide-react";
import {
  fetchScenarioPresets,
  createScenario,
  runScenario,
  fetchScenarioStatus,
  fetchScenarioReport,
} from "../services/api.js";

export default function ScenarioSimulationModal({
  isOpen,
  onClose,
  currentAOI,
  onApplyScenarioLayers,
}) {
  const [scenarioMode, setScenarioMode] = useState("custom_cyclone"); // 'live_analysis' | 'offline_demo' | 'custom_cyclone'
  const [scenarioPresets, setScenarioPresets] = useState([]);
  const [selectedPresetId, setSelectedPresetId] = useState("cyclone_associated_rainfall");

  // Geographic & Scenario Parameters
  const [scenarioName, setScenarioName] = useState("Hypothetical Cyclone — Sector Simulation");
  const [durationHours, setDurationHours] = useState(24);

  // Cyclone Parameters
  const [maxWindKmh, setMaxWindKmh] = useState(130);
  const [windGustKmh, setWindGustKmh] = useState(160);
  const [windDirectionDeg, setWindDirectionDeg] = useState(315);
  const [centralPressureHpa, setCentralPressureHpa] = useState(965);
  const [cycloneRadiusKm, setCycloneRadiusKm] = useState(65);
  const [movementSpeedKmh, setMovementSpeedKmh] = useState(18);

  // Rainfall Parameters
  const [rainfallIntensityMmHr, setRainfallIntensityMmHr] = useState(55);
  const [rainfallDurationHours, setRainfallDurationHours] = useState(16);
  const [rainfallDistribution, setRainfallDistribution] = useState("front_loaded");

  // Simulation Execution State
  const [activeScenarioId, setActiveScenarioId] = useState(null);
  const [simulationStatus, setSimulationStatus] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [selectedHour, setSelectedHour] = useState(0);

  const bbox = currentAOI?.bbox || [12.980, 80.190, 13.060, 80.290];
  const aoiName = currentAOI?.name || "Configured Study Area";

  // Dynamic Cyclone Category Calculation
  const getCycloneCategory = (windKmh) => {
    if (windKmh < 51) return { imd: "Depression", ss: "Tropical Depression", color: "text-blue-400" };
    if (windKmh < 62) return { imd: "Deep Depression", ss: "Tropical Depression", color: "text-cyan-400" };
    if (windKmh < 88) return { imd: "Cyclonic Storm (CS)", ss: "Tropical Storm", color: "text-emerald-400" };
    if (windKmh < 117) return { imd: "Severe Cyclonic Storm (SCS)", ss: "Tropical Storm", color: "text-amber-400" };
    if (windKmh < 166) return { imd: "Very Severe Cyclonic Storm (VSCS)", ss: "Category 1-2 Hurricane", color: "text-orange-400" };
    if (windKmh < 221) return { imd: "Extremely Severe Cyclonic Storm (ESCS)", ss: "Category 3-4 Major Hurricane", color: "text-rose-400" };
    return { imd: "Super Cyclonic Storm (SuCS)", ss: "Category 5 Major Hurricane", color: "text-purple-400" };
  };

  const cycloneCat = getCycloneCategory(maxWindKmh);

  useEffect(() => {
    if (!isOpen) return;
    loadPresets();
  }, [isOpen]);

  // Poll active scenario simulation
  useEffect(() => {
    if (!activeScenarioId || !isSimulating) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetchScenarioStatus(activeScenarioId);
        setSimulationStatus(res);
        if (res.status === "COMPLETED" || res.status === "FAILED") {
          setIsSimulating(false);
          if (res.snapshots?.length > 0 && onApplyScenarioLayers) {
            onApplyScenarioLayers(res.snapshots[0]);
          }
        }
      } catch (err) {
        setIsSimulating(false);
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [activeScenarioId, isSimulating]);

  const loadPresets = async () => {
    try {
      const presets = await fetchScenarioPresets();
      setScenarioPresets(presets);
    } catch (err) {
      console.warn("Failed to load scenario presets", err);
    }
  };

  const handleSelectPreset = (preset) => {
    setSelectedPresetId(preset.id);
    setScenarioName(preset.name);
    setDurationHours(preset.duration_hours || 24);
    if (preset.cyclone_params) {
      setMaxWindKmh(preset.cyclone_params.max_wind_speed_kmh || 100);
      setWindGustKmh(preset.cyclone_params.wind_gust_kmh || 120);
      setWindDirectionDeg(preset.cyclone_params.wind_direction_deg || 315);
      setCentralPressureHpa(preset.cyclone_params.central_pressure_hpa || 980);
      setCycloneRadiusKm(preset.cyclone_params.cyclone_radius_km || 50);
      setMovementSpeedKmh(preset.cyclone_params.movement_speed_kmh || 15);
    }
    if (preset.rainfall_params) {
      setRainfallIntensityMmHr(preset.rainfall_params.rainfall_intensity_mm_hr || 50);
      setRainfallDurationHours(preset.rainfall_params.rainfall_duration_hours || 12);
      setRainfallDistribution(preset.rainfall_params.rainfall_distribution || "front_loaded");
    }
  };

  const handleRunSimulation = async () => {
    setIsSimulating(true);
    setSimulationStatus(null);
    try {
      const createRes = await createScenario({
        name: scenarioName,
        bbox: bbox,
        cyclone_params: {
          max_wind_speed_kmh: Number(maxWindKmh),
          wind_gust_kmh: Number(windGustKmh),
          wind_direction_deg: Number(windDirectionDeg),
          central_pressure_hpa: Number(centralPressureHpa),
          cyclone_radius_km: Number(cycloneRadiusKm),
          movement_speed_kmh: Number(movementSpeedKmh),
        },
        rainfall_params: {
          rainfall_intensity_mm_hr: Number(rainfallIntensityMmHr),
          rainfall_duration_hours: Number(rainfallDurationHours),
          rainfall_distribution: rainfallDistribution,
        },
        duration_hours: Number(durationHours),
      });

      setActiveScenarioId(createRes.scenario_id);
      await runScenario(createRes.scenario_id);
    } catch (err) {
      console.error("Scenario run failed", err);
      setIsSimulating(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!activeScenarioId) return;
    try {
      const report = await fetchScenarioReport(activeScenarioId);
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `scenario_${activeScenarioId}_report.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
    }
  };

  const handleScrubTimeline = (hour) => {
    setSelectedHour(hour);
    const snap = simulationStatus?.snapshots?.find((s) => s.hour === hour);
    if (snap && onApplyScenarioLayers) {
      onApplyScenarioLayers(snap);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-md flex items-center justify-center p-4 overflow-y-auto">
      <div className="w-full max-w-5xl bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh]">
        {/* Header */}
        <div className="px-5 py-3.5 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/15 border border-cyan-500/40 text-cyan-400">
              <Wind className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-mono font-bold text-slate-100 uppercase tracking-wider">
                  Interactive Cyclone &amp; Extreme Rainfall Scenario Simulator
                </h2>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border bg-purple-500/20 text-purple-300 border-purple-500/40 font-bold uppercase">
                  VIVA SCENARIO LAB
                </span>
              </div>
              <p className="text-xs text-slate-400">
                AOI: <span className="text-slate-200 font-semibold">{aoiName}</span> | 2D Shallow-Water Numerical Flow Solver
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Mode Selector Pill */}
        <div className="px-5 py-2.5 bg-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="text-slate-400 text-[11px]">Simulation Mode:</span>
            {[
              { id: "live_analysis", label: "1. Live Disaster Analysis" },
              { id: "offline_demo", label: "2. Offline Demo Simulation" },
              { id: "custom_cyclone", label: "3. Custom Cyclone Scenario" },
            ].map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setScenarioMode(m.id)}
                className={`px-2.5 py-1 rounded text-[11px] font-semibold transition ${
                  scenarioMode === m.id
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold"
                    : "bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          <div className="text-[11px] text-amber-300/90 flex items-center gap-1.5">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            Hypothetical Disaster Model (Scientific Review &amp; Defense)
          </div>
        </div>

        {/* Modal Body */}
        <div className="flex-1 p-5 overflow-y-auto space-y-5 text-xs font-mono">
          {/* Preset Buttons */}
          <div className="space-y-2">
            <span className="text-[11px] text-slate-400 uppercase font-bold tracking-wider">
              Preconfigured Viva Demonstration Scenarios:
            </span>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
              {scenarioPresets.map((p) => {
                const isSelected = selectedPresetId === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => handleSelectPreset(p)}
                    className={`p-3 rounded-lg text-left transition border ${
                      isSelected
                        ? "bg-cyan-950/40 border-cyan-500/50 shadow-[0_0_12px_rgba(6,182,212,0.2)]"
                        : "bg-slate-950 border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    <div className="text-slate-200 font-bold text-xs">{p.name}</div>
                    <div className="text-[10px] text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                      {p.description}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Config Grid: Cyclone + Rainfall + Environment */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Left: Cyclone Vortex Parameters */}
            <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-3.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Wind className="w-4 h-4 text-cyan-400" />
                  <span className="font-bold text-slate-200 uppercase tracking-wider text-xs">
                    Cyclone Wind &amp; Vortex Dynamics
                  </span>
                </div>
                {/* Dynamic Category Badge */}
                <span className={`text-[10px] px-2 py-0.5 rounded font-bold border border-current ${cycloneCat.color}`}>
                  {cycloneCat.imd} ({cycloneCat.ss})
                </span>
              </div>

              {/* Max Sustained Wind Speed Slider */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-slate-400">Max Sustained Wind Speed:</span>
                  <span className="text-cyan-300 font-bold">{maxWindKmh} km/h</span>
                </div>
                <input
                  type="range"
                  min="30"
                  max="260"
                  step="5"
                  value={maxWindKmh}
                  onChange={(e) => setMaxWindKmh(Number(e.target.value))}
                  className="w-full accent-cyan-400 h-1.5 bg-slate-900 rounded"
                />
              </div>

              {/* Wind Gust & Central Pressure */}
              <div className="grid grid-cols-2 gap-3 text-[11px]">
                <div>
                  <label className="text-[10px] text-slate-500 uppercase">Wind Gust (km/h)</label>
                  <input
                    type="number"
                    value={windGustKmh}
                    onChange={(e) => setWindGustKmh(e.target.value)}
                    className="w-full mt-1 bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-slate-500 uppercase">Central Pressure (hPa)</label>
                  <input
                    type="number"
                    value={centralPressureHpa}
                    onChange={(e) => setCentralPressureHpa(e.target.value)}
                    className="w-full mt-1 bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200"
                  />
                </div>
              </div>

              {/* Cyclone Radius & Forward Speed */}
              <div className="grid grid-cols-2 gap-3 text-[11px]">
                <div>
                  <label className="text-[10px] text-slate-500 uppercase">Vortex Radius (km)</label>
                  <input
                    type="number"
                    value={cycloneRadiusKm}
                    onChange={(e) => setCycloneRadiusKm(e.target.value)}
                    className="w-full mt-1 bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-slate-500 uppercase">Forward Speed (km/h)</label>
                  <input
                    type="number"
                    value={movementSpeedKmh}
                    onChange={(e) => setMovementSpeedKmh(e.target.value)}
                    className="w-full mt-1 bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200"
                  />
                </div>
              </div>
            </div>

            {/* Right: Extreme Rainfall & Temporal Distribution */}
            <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-3.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CloudRain className="w-4 h-4 text-emerald-400" />
                  <span className="font-bold text-slate-200 uppercase tracking-wider text-xs">
                    Extreme Rainfall Forcing (mm/h)
                  </span>
                </div>
                <span className="text-[10px] px-2 py-0.5 rounded font-bold bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
                  {rainfallDistribution.replace("_", " ").toUpperCase()}
                </span>
              </div>

              {/* Rainfall Intensity Slider */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-slate-400">Peak Rainfall Intensity:</span>
                  <span className="text-emerald-400 font-bold">{rainfallIntensityMmHr} mm/h</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="140"
                  step="5"
                  value={rainfallIntensityMmHr}
                  onChange={(e) => setRainfallIntensityMmHr(Number(e.target.value))}
                  className="w-full accent-emerald-400 h-1.5 bg-slate-900 rounded"
                />
              </div>

              {/* Rainfall Duration & Temporal Distribution */}
              <div className="grid grid-cols-2 gap-3 text-[11px]">
                <div>
                  <label className="text-[10px] text-slate-500 uppercase">Rainfall Duration (Hours)</label>
                  <input
                    type="number"
                    value={rainfallDurationHours}
                    onChange={(e) => setRainfallDurationHours(e.target.value)}
                    className="w-full mt-1 bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-slate-500 uppercase">Temporal Distribution</label>
                  <select
                    value={rainfallDistribution}
                    onChange={(e) => setRainfallDistribution(e.target.value)}
                    className="w-full mt-1 bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200"
                  >
                    <option value="front_loaded">Front-Loaded (Cloudburst Surge)</option>
                    <option value="back_loaded">Back-Loaded (Delayed Peak)</option>
                    <option value="uniform">Uniform Steady Inflow</option>
                    <option value="time_series">Vortex Spiral Rainbands</option>
                  </select>
                </div>
              </div>

              {/* Simulation Horizon Select */}
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase">Simulation Horizon Duration</label>
                <div className="grid grid-cols-4 gap-2">
                  {[6, 12, 24, 48].map((h) => (
                    <button
                      key={h}
                      type="button"
                      onClick={() => setDurationHours(h)}
                      className={`py-1 text-center rounded border text-xs font-bold transition ${
                        durationHours === h
                          ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                          : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
                      }`}
                    >
                      {h} Hours
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Action Row */}
          <div className="flex items-center justify-between p-3.5 rounded-xl bg-slate-950 border border-slate-800">
            <div className="text-[11px] text-slate-400">
              Hydro Solver: <strong className="text-slate-200">2D Diffusive Wave (Manning n=0.035, Horton Infiltration)</strong>
            </div>
            <button
              type="button"
              onClick={handleRunSimulation}
              disabled={isSimulating}
              className="px-5 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold transition flex items-center gap-2 shadow-[0_0_15px_rgba(6,182,212,0.35)] disabled:opacity-50"
            >
              {isSimulating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
              {isSimulating ? "Solving 2D Shallow-Water Fluid Equations..." : "Run Scenario Simulation"}
            </button>
          </div>

          {/* Simulation Results & Timeline Player */}
          {simulationStatus?.results && (
            <div className="p-4 rounded-xl bg-slate-950/90 border border-emerald-500/40 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span className="font-bold text-slate-100 text-xs uppercase">
                    Simulation Solved: {simulationStatus.results.name}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={handleDownloadReport}
                  className="px-3 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs font-bold transition flex items-center gap-1.5"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download Scenario Report (JSON)
                </button>
              </div>

              {/* Analytical Telemetry Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5 font-mono text-xs">
                <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Inundated Area</div>
                  <div className="text-sm font-bold text-rose-400">
                    {simulationStatus.results.final_flooded_km2} km²
                  </div>
                </div>
                <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Max Water Depth</div>
                  <div className="text-sm font-bold text-cyan-300">
                    {simulationStatus.results.max_flood_depth_m} m
                  </div>
                </div>
                <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Exposed Buildings</div>
                  <div className="text-sm font-bold text-amber-400">
                    {simulationStatus.results.submerged_buildings} structures
                  </div>
                </div>
                <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Mass Conservation</div>
                  <div className="text-sm font-bold text-emerald-400">
                    {simulationStatus.results.mass_balance_error_pct.toFixed(2)}% Error
                  </div>
                </div>
              </div>

              {/* Timeline Scrubber */}
              {simulationStatus.snapshots?.length > 0 && (
                <div className="space-y-2 pt-2 border-t border-slate-800">
                  <div className="flex justify-between text-xs text-slate-400">
                    <span>Timeline Progression Scrub:</span>
                    <strong className="text-cyan-300">T+{selectedHour}:00 Hours</strong>
                  </div>
                  <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
                    {simulationStatus.snapshots.map((s) => (
                      <button
                        key={s.hour}
                        type="button"
                        onClick={() => handleScrubTimeline(s.hour)}
                        className={`px-3 py-1.5 rounded text-xs font-bold transition shrink-0 ${
                          selectedHour === s.hour
                            ? "bg-cyan-500 text-slate-950 shadow-[0_0_8px_rgba(6,182,212,0.4)]"
                            : "bg-slate-900 text-slate-300 border border-slate-800 hover:text-slate-100"
                        }`}
                      >
                        {s.label} ({s.flooded_area_km2} km²)
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3.5 bg-slate-950 border-t border-slate-800 flex items-center justify-between text-xs font-mono">
          <span className="text-[11px] text-slate-500">
            Scientifically defensible 2D diffusive wave equation &amp; Rankine vortex wind field
          </span>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-200 text-xs font-semibold border border-slate-800 transition"
          >
            Close Lab
          </button>
        </div>
      </div>
    </div>
  );
}

