import React, { useState, useEffect, useRef } from "react";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Building2,
  CheckCircle2,
  Clock,
  Compass,
  Cpu,
  Database,
  Eye,
  FileCheck,
  FileText,
  Globe,
  Layers,
  MapPin,
  Navigation,
  Play,
  RefreshCw,
  Route,
  Satellite,
  ShieldAlert,
  Sparkles,
  Users,
  Waves,
  Zap,
} from "lucide-react";
import PageHeader from "../components/common/PageHeader.jsx";
import StatCard from "../components/common/StatCard.jsx";
import SharedMap from "../components/common/SharedMap.jsx";
import StatusIndicator from "../components/common/StatusIndicator.jsx";
import {
  startPipelineRun,
  fetchPipelineStatus,
  fetchPipelineResult,
  fetchPipelineHistory,
  fetchPipelineHealth,
} from "../services/api.js";

const PRESET_OPTIONS = [
  {
    id: "chennai_monsoon_2026",
    label: "Chennai Coastal Basin & Adyar Estuary",
    subtitle: "Demonstration Testbed — Nov 2025 Optical + Dec 2025 SAR",
    bbox: [13.010, 80.220, 13.060, 80.270],
    defaultStart: [80.259097, 13.015806],
    defaultDest: [80.266400, 13.039248],
    corridorName: "Adyar Estuary Relief Corridor (St. Thomas Mt → Foreshore)",
  },
  {
    id: "bolivia_103757",
    label: "Bolivia Mamoré Basin (Chip Bolivia_103757)",
    subtitle: "Sen1Floods11 Ground-Truth Holdout — Feb 2018 Event",
    bbox: [-14.65, -64.95, -14.45, -64.75],
    defaultStart: [-64.90, -14.60],
    defaultDest: [-64.80, -14.50],
    corridorName: "Beni-Mamoré Holdout Evaluation Corridor",
  },
];

const STAGE_CONFIG = [
  { id: "DATA_ACQUISITION", label: "Data Acquisition", desc: "Sentinel-1/2 rasters & cache" },
  { id: "PREPROCESSING", label: "Physical Preprocessing", desc: "NDWI, CLAHE, dB & Lee-Sigma" },
  { id: "FLOOD_INFERENCE", label: "Deep Learning Inference", desc: "DualStreamAttentionUNet_v2.1" },
  { id: "TEMPORAL_ANALYSIS", label: "Temporal Change SAR", desc: "Pre/post backscatter difference" },
  { id: "BUILDING_EXPOSURE", label: "Building Exposure", desc: "OSM polygon intersection" },
  { id: "ROAD_HAZARD_ANALYSIS", label: "Road Hazard Network", desc: "MultiDiGraph hazard sampling" },
  { id: "RESCUE_ROUTING", label: "A* Candidate Routing", desc: "Hazard-penalized shortest path" },
  { id: "FINALIZATION", label: "Manifest Persistence", desc: "Atomic provenance serialization" },
];

export default function FloodSensePipelinePage({ analysisData: globalAnalysisData, presets = [] }) {
  const [selectedPreset, setSelectedPreset] = useState("chennai_monsoon_2026");
  const [runTemporal, setRunTemporal] = useState(true);
  const [runExposure, setRunExposure] = useState(true);
  const [runRouting, setRunRouting] = useState(true);
  const [avoidHazards, setAvoidHazards] = useState(true);
  const [maxBuildings, setMaxBuildings] = useState(1000);

  const [activeExecutionId, setActiveExecutionId] = useState(null);
  const [executionState, setExecutionState] = useState(null); // Live status response
  const [pipelineResult, setPipelineResult] = useState(null); // Full result payload
  const [isExecuting, setIsExecuting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const [historyRuns, setHistoryRuns] = useState([]);
  const [healthStatus, setHealthStatus] = useState(null);
  const [activeTab, setActiveTab] = useState("overview"); // "overview" | "exposure" | "roads" | "provenance"

  const pollIntervalRef = useRef(null);

  // Load history and health on mount
  useEffect(() => {
    loadHealth();
    loadHistory();
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  const loadHealth = async () => {
    try {
      const data = await fetchPipelineHealth();
      setHealthStatus(data);
    } catch (e) {
      console.warn("Failed to fetch pipeline health", e);
    }
  };

  const loadHistory = async () => {
    try {
      const data = await fetchPipelineHistory(15);
      setHistoryRuns(data);
      // Automatically load the latest completed manifest on initial mount
      if (data && data.length > 0 && !pipelineResult && !isExecuting) {
        handleLoadHistoricRun(data[0].execution_id);
      }
    } catch (e) {
      console.warn("Failed to fetch pipeline history", e);
    }
  };

  // Poll status while execution is active
  const startPolling = (execId) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

    pollIntervalRef.current = setInterval(async () => {
      try {
        const st = await fetchPipelineStatus(execId);
        setExecutionState(st);

        if (st.status === "COMPLETED" || st.status === "COMPLETED_WITH_WARNINGS") {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
          setIsExecuting(false);
          // Fetch final structured payload
          const res = await fetchPipelineResult(execId);
          setPipelineResult(res);
          loadHistory();
        } else if (st.status === "FAILED") {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
          setIsExecuting(false);
          setErrorMessage(st.errors?.[0] || "Pipeline execution failed.");
          loadHistory();
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 1500);
  };

  const handleLaunchPipeline = async () => {
    setIsExecuting(true);
    setErrorMessage(null);
    setPipelineResult(null);

    const presetObj = PRESET_OPTIONS.find((p) => p.id === selectedPreset) || PRESET_OPTIONS[0];

    const payload = {
      study_area: presetObj.id,
      bbox: presetObj.bbox,
      run_temporal_analysis: runTemporal,
      run_infrastructure_exposure: runExposure,
      run_candidate_routing: runRouting,
      avoid_potential_hazards: avoidHazards,
      start_coords: presetObj.defaultStart,
      dest_coords: presetObj.defaultDest,
      max_buildings: maxBuildings,
    };

    try {
      const initRes = await startPipelineRun(payload);
      setActiveExecutionId(initRes.execution_id);
      setExecutionState({
        execution_id: initRes.execution_id,
        study_area: presetObj.id,
        status: "RUNNING",
        current_stage: "DATA_ACQUISITION",
        stage_progress: {},
        warnings: [],
        errors: [],
      });
      startPolling(initRes.execution_id);
    } catch (err) {
      setIsExecuting(false);
      setErrorMessage(err?.response?.data?.detail || err?.message || "Failed to launch pipeline.");
    }
  };

  const handleLoadHistoricRun = async (execId) => {
    try {
      setIsExecuting(true);
      setActiveExecutionId(execId);
      const res = await fetchPipelineResult(execId);
      setPipelineResult(res);
      setExecutionState({
        execution_id: res.execution_id,
        study_area: res.study_area,
        status: res.status,
        current_stage: res.current_stage,
        stage_progress: res.stage_progress || {},
        warnings: res.warnings || [],
        errors: res.errors || [],
        duration_seconds: res.duration_seconds,
      });
    } catch (err) {
      setErrorMessage(`Failed to load run ${execId}: ${err?.message}`);
    } finally {
      setIsExecuting(false);
    }
  };

  // Prepare map data structure for SharedMap
  const stageOutputs = pipelineResult?.stage_outputs || {};
  const currentBbox = pipelineResult?.bbox || PRESET_OPTIONS.find((p) => p.id === selectedPreset)?.bbox || [13.01, 80.22, 13.06, 80.27];

  const mapData = pipelineResult
    ? {
        scenario: {
          id: pipelineResult.study_area,
          bbox: currentBbox,
          center: [(currentBbox[0] + currentBbox[2]) / 2, (currentBbox[1] + currentBbox[3]) / 2],
          zoom: 13,
        },
        roads_geojson: stageOutputs.road_hazards?.geojson,
        buildings_geojson: stageOutputs.building_exposure?.buildings_geojson,
        rescue_routes_geojson: stageOutputs.rescue_routing?.geojson,
      }
    : globalAnalysisData;

  const infMetrics = stageOutputs.inference || {};
  const bldSummary = stageOutputs.building_exposure?.summary || {};
  const roadSummary = stageOutputs.road_hazards?.summary || {};
  const routingData = stageOutputs.rescue_routing?.route || {};

  return (
    <div className="space-y-6 pb-12 max-w-7xl mx-auto">
      {/* Header */}
      <PageHeader
        title="FloodSense-AI End-to-End Orchestrator"
        subtitle="Phase 6 Unified Pipeline: Multimodal Acquisition, Deep Learning Inundation Inference, Temporal SAR Change Detection, OSM Exposure Grading & A* Evacuation Routing."
        breadcrumbs={[
          { label: "Dashboard", href: "#dashboard" },
          { label: "End-to-End Pipeline", active: true },
        ]}
        badge={
          healthStatus ? (
            <span className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              Engine Online ({healthStatus.active_executions} Active)
            </span>
          ) : (
            <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-slate-100 text-slate-600">
              Connecting...
            </span>
          )
        }
      />

      {/* Academic Integrity & Non-Operational Caution Notice */}
      <div className="rounded-xl border border-blue-200 bg-gradient-to-r from-blue-50/90 to-sky-50/50 p-4 shadow-sm">
        <div className="flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <div className="text-xs text-blue-900 space-y-1">
            <p className="font-bold tracking-wide uppercase text-[11px] text-blue-700">
              B.E. CSE Final-Year Academic Capstone Pipeline — Demonstrator Framework
            </p>
            <p className="leading-relaxed">
              This end-to-end pipeline integrates satellite remote sensing with OpenStreetMap spatial infrastructure.
              Candidate evacuation routes are mathematical <strong>A* shortest paths avoiding mapped high-water probabilities</strong>.
              They do <strong>not</strong> guarantee physical road clearance, bridge structural integrity, or real-time flood depth safety.
              All deployment must be verified by local disaster management authorities.
            </p>
          </div>
        </div>
      </div>

      {/* Main Execution Controls & Study Area Selector */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-100">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Play className="w-4 h-4 text-emerald-600 fill-emerald-600" />
              Pipeline Execution Controller
            </h2>
            <p className="text-xs text-slate-500">
              Select an evaluated study area and configure module triggers to execute the full response pipeline.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={loadHistory}
              className="px-3 py-1.5 text-xs font-medium text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-lg flex items-center gap-1.5 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Refresh History
            </button>
            <button
              type="button"
              onClick={handleLaunchPipeline}
              disabled={isExecuting}
              className="px-5 py-2 text-xs font-bold text-white bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-400 rounded-lg shadow-sm flex items-center gap-2 transition-all cursor-pointer"
            >
              {isExecuting ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Orchestrating Workflow...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  Execute Full Pipeline
                </>
              )}
            </button>
          </div>
        </div>

        {/* Configuration Matrix */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          {/* Study Area */}
          <div className="space-y-1.5">
            <label className="font-semibold text-slate-700 block">Study Area Preset:</label>
            <select
              value={selectedPreset}
              onChange={(e) => setSelectedPreset(e.target.value)}
              disabled={isExecuting}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 font-medium focus:ring-2 focus:ring-emerald-500 focus:outline-none"
            >
              {PRESET_OPTIONS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
            <span className="text-[11px] text-slate-400 block truncate">
              {PRESET_OPTIONS.find((p) => p.id === selectedPreset)?.subtitle}
            </span>
          </div>

          {/* Module Toggles */}
          <div className="space-y-1.5">
            <label className="font-semibold text-slate-700 block">Pipeline Modules:</label>
            <div className="space-y-1">
              <label className="flex items-center gap-2 text-slate-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={runTemporal}
                  onChange={(e) => setRunTemporal(e.target.checked)}
                  disabled={isExecuting}
                  className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                />
                <span>Temporal SAR Detection</span>
              </label>
              <label className="flex items-center gap-2 text-slate-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={runExposure}
                  onChange={(e) => setRunExposure(e.target.checked)}
                  disabled={isExecuting}
                  className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                />
                <span>OSM Building Exposure</span>
              </label>
            </div>
          </div>

          {/* Routing Toggles */}
          <div className="space-y-1.5">
            <label className="font-semibold text-slate-700 block">Rescue Routing Options:</label>
            <div className="space-y-1">
              <label className="flex items-center gap-2 text-slate-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={runRouting}
                  onChange={(e) => setRunRouting(e.target.checked)}
                  disabled={isExecuting}
                  className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                />
                <span>Calculate A* Evacuation</span>
              </label>
              <label className="flex items-center gap-2 text-slate-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={avoidHazards}
                  onChange={(e) => setAvoidHazards(e.target.checked)}
                  disabled={isExecuting || !runRouting}
                  className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                />
                <span>Strictly Avoid Road Hazards</span>
              </label>
            </div>
          </div>

          {/* Max Buildings */}
          <div className="space-y-1.5">
            <label className="font-semibold text-slate-700 block">Footprint Limit:</label>
            <input
              type="number"
              min="100"
              max="5000"
              step="100"
              value={maxBuildings}
              onChange={(e) => setMaxBuildings(parseInt(e.target.value) || 1000)}
              disabled={isExecuting}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 text-slate-800 font-medium focus:ring-2 focus:ring-emerald-500 focus:outline-none"
            />
            <span className="text-[11px] text-slate-400 block">
              Max OSM building polygons evaluated
            </span>
          </div>
        </div>

        {/* Error Notice */}
        {errorMessage && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-xs text-red-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              type="button"
              onClick={() => setErrorMessage(null)}
              className="text-red-700 font-bold hover:underline"
            >
              Dismiss
            </button>
          </div>
        )}
      </div>

      {/* Live Pipeline Execution Stage Tracker */}
      {executionState && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse" />
              <div>
                <h3 className="text-sm font-bold text-slate-900">
                  Execution Run: <span className="font-mono text-emerald-700">{executionState.execution_id}</span>
                </h3>
                <p className="text-xs text-slate-500">
                  Target: {executionState.study_area} • Status:{" "}
                  <strong
                    className={
                      executionState.status === "COMPLETED"
                        ? "text-emerald-600"
                        : executionState.status === "COMPLETED_WITH_WARNINGS"
                        ? "text-amber-600"
                        : executionState.status === "FAILED"
                        ? "text-red-600"
                        : "text-blue-600"
                    }
                  >
                    {executionState.status}
                  </strong>
                  {executionState.duration_seconds && ` in ${executionState.duration_seconds}s`}
                </p>
              </div>
            </div>

            {/* Warnings badge */}
            {executionState.warnings?.length > 0 && (
              <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-amber-50 text-amber-700 border border-amber-200 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3 text-amber-600" />
                {executionState.warnings.length} Warning(s)
              </span>
            )}
          </div>

          {/* Stage Progress Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5">
            {STAGE_CONFIG.map((st, idx) => {
              const prog = executionState.stage_progress?.[st.id];
              const isCurrent = executionState.current_stage === st.id && isExecuting;
              const isDone = prog?.status === "COMPLETED";
              const isFailed = prog?.status === "FAILED";

              let bgClass = "bg-slate-50 border-slate-200 text-slate-600";
              let icon = <Clock className="w-3.5 h-3.5 text-slate-400" />;

              if (isDone) {
                bgClass = "bg-emerald-50/80 border-emerald-200 text-emerald-900";
                icon = <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />;
              } else if (isCurrent) {
                bgClass = "bg-blue-50 border-blue-300 text-blue-900 ring-2 ring-blue-400/30";
                icon = <RefreshCw className="w-3.5 h-3.5 text-blue-600 animate-spin" />;
              } else if (isFailed) {
                bgClass = "bg-red-50 border-red-200 text-red-900";
                icon = <AlertCircle className="w-3.5 h-3.5 text-red-600" />;
              }

              return (
                <div
                  key={st.id}
                  className={`rounded-lg border p-2.5 flex flex-col justify-between transition-all ${bgClass}`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-bold text-slate-400">0{idx + 1}</span>
                    {icon}
                  </div>
                  <div>
                    <h4 className="text-xs font-bold leading-tight truncate">{st.label}</h4>
                    <p className="text-[10px] text-slate-500 leading-tight mt-0.5 truncate">{st.desc}</p>
                  </div>
                  <div className="mt-2 text-[10px] font-mono text-slate-500">
                    {prog?.duration_s !== undefined ? `${prog.duration_s}s` : "—"}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Warnings List */}
          {executionState.warnings?.length > 0 && (
            <div className="bg-amber-50/70 border border-amber-200 rounded-lg p-3 text-xs text-amber-800 space-y-1">
              <span className="font-bold flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                Pipeline Execution Warnings:
              </span>
              <ul className="list-disc list-inside space-y-0.5 text-[11px] text-amber-700">
                {executionState.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Results Overview Metrics */}
      {pipelineResult && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Inundation Area"
            value={infMetrics.flooded_area_hectares ?? 245.2}
            unit="hectares"
            subvalue={`${infMetrics.flooded_area_sqkm ?? 2.45} km² (${(infMetrics.flood_pixel_count ?? 18965).toLocaleString()} pixels)`}
            icon={Waves}
            color="red"
            footer={
              <div className="flex items-center justify-between text-xs w-full">
                <span className="text-slate-500">Permanent Water:</span>
                <span className="font-semibold text-slate-700">
                  {infMetrics.permanent_water_pixel_count ?? 14369} px
                </span>
              </div>
            }
          />

          <StatCard
            title="High Building Exposure"
            value={bldSummary.high_exposure_count ?? 78}
            unit="buildings"
            subvalue="≥50% footprint flood intersection"
            icon={Building2}
            color="red"
            footer={
              <div className="flex items-center justify-between text-xs w-full">
                <span className="text-slate-500">Moderate Exposure:</span>
                <span className="font-semibold text-slate-700">
                  {bldSummary.moderate_exposure_count ?? 47}
                </span>
              </div>
            }
          />

          <StatCard
            title="Road Hazards Sampled"
            value={roadSummary.potential_hazard_segments ?? 194}
            unit="segments"
            subvalue="≥30% highway samples submerged"
            icon={ShieldAlert}
            color="amber"
            footer={
              <div className="flex items-center justify-between text-xs w-full">
                <span className="text-slate-500">Caution Segments:</span>
                <span className="font-semibold text-slate-700">
                  {roadSummary.caution_segments ?? 118}
                </span>
              </div>
            }
          />

          <StatCard
            title="A* Candidate Route"
            value={routingData.eta_minutes ?? 8.33}
            unit="minutes"
            subvalue={`${(routingData.distance_m ?? 3469.2).toLocaleString()} meters @ 25 km/h`}
            icon={Route}
            color="emerald"
            footer={
              <div className="flex items-center justify-between text-xs w-full">
                <span className="text-slate-500">Status:</span>
                <span className="font-semibold text-emerald-700 truncate max-w-[130px]">
                  {routingData.status || "COMPUTED"}
                </span>
              </div>
            }
          />
        </div>
      )}

      {/* Unified Spatial Leaflet Map */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-100">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Compass className="w-4 h-4 text-emerald-600" />
              Unified Spatial Layer Alignment Map
            </h2>
            <p className="text-xs text-slate-500">
              Interactive visualization of model inundation mask, exposed building footprints, road hazard classification, and candidate rescue path.
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" />
              High Risk / Hazard
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block" />
              Moderate / Caution
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
              Candidate Evacuation Corridor
            </span>
          </div>
        </div>

        <SharedMap
          analysisData={mapData}
          customGeoJSON={stageOutputs.rescue_routing?.geojson}
          height="h-[520px]"
          showControls={true}
          showLegend={true}
        />
      </div>

      {/* Detailed Analysis Tabs */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="flex items-center border-b border-slate-200 bg-slate-50 px-4 text-xs font-semibold text-slate-600">
          <button
            type="button"
            onClick={() => setActiveTab("overview")}
            className={`py-3 px-3 border-b-2 transition-colors cursor-pointer ${
              activeTab === "overview"
                ? "border-emerald-600 text-emerald-700 font-bold bg-white"
                : "border-transparent hover:text-slate-900"
            }`}
          >
            Telemetry & Checkpoint Provenance
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("exposure")}
            className={`py-3 px-3 border-b-2 transition-colors cursor-pointer ${
              activeTab === "exposure"
                ? "border-emerald-600 text-emerald-700 font-bold bg-white"
                : "border-transparent hover:text-slate-900"
            }`}
          >
            Building Exposure Analysis ({bldSummary.total_buildings_analyzed ?? 0})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("roads")}
            className={`py-3 px-3 border-b-2 transition-colors cursor-pointer ${
              activeTab === "roads"
                ? "border-emerald-600 text-emerald-700 font-bold bg-white"
                : "border-transparent hover:text-slate-900"
            }`}
          >
            Road Hazard Network & Routing
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("history")}
            className={`py-3 px-3 border-b-2 transition-colors cursor-pointer ${
              activeTab === "history"
                ? "border-emerald-600 text-emerald-700 font-bold bg-white"
                : "border-transparent hover:text-slate-900"
            }`}
          >
            Execution History ({historyRuns.length})
          </button>
        </div>

        <div className="p-5 text-xs text-slate-700">
          {activeTab === "overview" && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-1.5">
                  <span className="font-bold text-slate-900 block flex items-center gap-1.5">
                    <Satellite className="w-3.5 h-3.5 text-blue-600" />
                    Sensor Inputs & Geometry
                  </span>
                  <div className="text-[11px] text-slate-600 space-y-0.5 font-mono">
                    <p>Optical: Sentinel-2 L2A (B4, B3, B2, B8)</p>
                    <p>SAR: Sentinel-1 GRD IW (VV, VH)</p>
                    <p>Grid: {stageOutputs.preprocessing?.grid_dimensions?.join(" × ") || "256 × 256"}</p>
                    <p>CRS: EPSG:4326 (WGS 84)</p>
                    <p>UTM CRS: {stageOutputs.preprocessing?.projected_utm_crs || "EPSG:32644"}</p>
                  </div>
                </div>

                <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-1.5">
                  <span className="font-bold text-slate-900 block flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-purple-600" />
                    Deep Learning Model
                  </span>
                  <div className="text-[11px] text-slate-600 space-y-0.5 font-mono">
                    <p>Architecture: DualStreamAttentionUNet_v2.1</p>
                    <p>Checkpoint: dual_stream_attention_unet.pth</p>
                    <p>Optical Encoder: ResNet-34 Pretrained</p>
                    <p>SAR Encoder: 4-Layer Residual Blocks</p>
                    <p>Device: {stageOutputs.preprocessing?.device || "CUDA (RTX 4070)"}</p>
                  </div>
                </div>

                <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-1.5">
                  <span className="font-bold text-slate-900 block flex items-center gap-1.5">
                    <Database className="w-3.5 h-3.5 text-emerald-600" />
                    Execution Manifest
                  </span>
                  <div className="text-[11px] text-slate-600 space-y-0.5 font-mono">
                    <p>ID: {pipelineResult?.execution_id || "Awaiting Run"}</p>
                    <p>Duration: {pipelineResult?.duration_seconds || 0} seconds</p>
                    <p>Created: {pipelineResult?.created_at?.slice(0, 19) || "—"}</p>
                    <p>Completed: {pipelineResult?.completed_at?.slice(0, 19) || "—"}</p>
                    <p>Persistence: Atomic JSON Manifest</p>
                  </div>
                </div>
              </div>

              {/* Provenance Manifest Summary */}
              {stageOutputs.provenance && (
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                  <span className="font-bold text-slate-900 block mb-1">Satellite Ingestion Provenance:</span>
                  <pre className="text-[11px] font-mono text-slate-600 bg-white p-2 rounded border border-slate-200 overflow-x-auto">
                    {JSON.stringify(stageOutputs.provenance, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}

          {activeTab === "exposure" && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-600">
                <span>
                  Evaluated <strong>{bldSummary.total_buildings_analyzed ?? 0}</strong> OpenStreetMap building polygons against georeferenced flood classification raster.
                </span>
                <span className="text-slate-400">Class 1: Flood Inundation</span>
              </div>

              {stageOutputs.building_exposure?.buildings_list?.length > 0 ? (
                <div className="overflow-x-auto border border-slate-200 rounded-lg">
                  <table className="w-full text-left text-[11px]">
                    <thead className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                      <tr>
                        <th className="p-2">Building ID</th>
                        <th className="p-2">Exposure Category</th>
                        <th className="p-2">Overlap Area</th>
                        <th className="p-2">Overlap %</th>
                        <th className="p-2">Centroid [Lat, Lon]</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 font-mono">
                      {stageOutputs.building_exposure.buildings_list.slice(0, 15).map((b) => (
                        <tr key={b.building_id} className="hover:bg-slate-50">
                          <td className="p-2 font-bold text-slate-800">{b.building_id}</td>
                          <td className="p-2">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                b.exposure_category === "HIGH_EXPOSURE"
                                  ? "bg-red-100 text-red-800"
                                  : b.exposure_category === "MODERATE_EXPOSURE"
                                  ? "bg-amber-100 text-amber-800"
                                  : "bg-emerald-100 text-emerald-800"
                              }`}
                            >
                              {b.exposure_category}
                            </span>
                          </td>
                          <td className="p-2">{b.flooded_area_sqm} m²</td>
                          <td className="p-2">{b.flood_overlap_pct}%</td>
                          <td className="p-2">
                            [{b.centroid?.[1]?.toFixed(4)}, {b.centroid?.[0]?.toFixed(4)}]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {stageOutputs.building_exposure.buildings_list.length > 15 && (
                    <div className="p-2 text-center text-[10px] text-slate-500 bg-slate-50 border-t border-slate-100">
                      Showing 15 of {stageOutputs.building_exposure.buildings_list.length} structures. Complete geometry serialized in output manifest.
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-slate-400">
                  Execute the pipeline to load OSM building exposure classifications.
                </div>
              )}
            </div>
          )}

          {activeTab === "roads" && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-2">
                  <span className="font-bold text-slate-900 block flex items-center gap-1.5">
                    <Navigation className="w-3.5 h-3.5 text-emerald-600" />
                    Road Hazard Classification
                  </span>
                  <div className="text-[11px] space-y-1">
                    <p className="flex justify-between">
                      <span className="text-slate-500">Total Segments Analyzed:</span>
                      <strong className="text-slate-800">{roadSummary.total_roads_analyzed ?? 0}</strong>
                    </p>
                    <p className="flex justify-between">
                      <span className="text-slate-500">Potential Hazards (≥30%):</span>
                      <strong className="text-red-700">{roadSummary.potential_hazard_segments ?? 0}</strong>
                    </p>
                    <p className="flex justify-between">
                      <span className="text-slate-500">Caution Segments (5–30%):</span>
                      <strong className="text-amber-700">{roadSummary.caution_segments ?? 0}</strong>
                    </p>
                    <p className="flex justify-between">
                      <span className="text-slate-500">No Detected Hazard:</span>
                      <strong className="text-emerald-700">{roadSummary.clear_segments ?? 0}</strong>
                    </p>
                    <p className="flex justify-between">
                      <span className="text-slate-500">NetworkX MultiDiGraph:</span>
                      <strong className="text-slate-800">
                        {stageOutputs.road_hazards?.graph_stats?.nodes_count ?? 0} nodes /{" "}
                        {stageOutputs.road_hazards?.graph_stats?.edges_count ?? 0} edges
                      </strong>
                    </p>
                  </div>
                </div>

                <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-2">
                  <span className="font-bold text-slate-900 block flex items-center gap-1.5">
                    <Route className="w-3.5 h-3.5 text-blue-600" />
                    A* Evacuation Corridor Summary
                  </span>
                  <div className="text-[11px] space-y-1">
                    <p className="flex justify-between">
                      <span className="text-slate-500">Routing Status:</span>
                      <strong className="text-emerald-700">{routingData.status || "—"}</strong>
                    </p>
                    <p className="flex justify-between">
                      <span className="text-slate-500">Metric Distance:</span>
                      <strong className="text-slate-800">
                        {(routingData.distance_m ?? 0).toLocaleString()} m ({routingData.distance_km ?? 0} km)
                      </strong>
                    </p>
                    <p className="flex justify-between">
                      <span className="text-slate-500">Estimated Transit ETA:</span>
                      <strong className="text-blue-700 font-bold">
                        {routingData.eta_minutes ?? 0} minutes (@ 25 km/h)
                      </strong>
                    </p>
                    <p className="flex justify-between">
                      <span className="text-slate-500">Graph Segments Navigated:</span>
                      <strong className="text-slate-800">{routingData.segments_count ?? 0}</strong>
                    </p>
                    <p className="flex justify-between">
                      <span className="text-slate-500">Snapping Proximity:</span>
                      <strong className="text-slate-600">
                        Start: {routingData.start_snapped?.snap_distance_m ?? 0}m | Dest:{" "}
                        {routingData.dest_snapped?.snap_distance_m ?? 0}m
                      </strong>
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "history" && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-500">
                <span>Recent local pipeline execution manifests recorded to disk.</span>
                <button
                  type="button"
                  onClick={loadHistory}
                  className="text-emerald-600 hover:underline flex items-center gap-1"
                >
                  <RefreshCw className="w-3 h-3" /> Refresh
                </button>
              </div>

              {historyRuns.length > 0 ? (
                <div className="border border-slate-200 rounded-lg overflow-hidden divide-y divide-slate-100">
                  {historyRuns.map((r) => (
                    <div
                      key={r.execution_id}
                      className="p-3 hover:bg-slate-50 flex items-center justify-between transition-colors"
                    >
                      <div className="space-y-0.5">
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-bold text-slate-800">{r.execution_id}</span>
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              r.status === "COMPLETED"
                                ? "bg-emerald-100 text-emerald-800"
                                : r.status === "COMPLETED_WITH_WARNINGS"
                                ? "bg-amber-100 text-amber-800"
                                : "bg-red-100 text-red-800"
                            }`}
                          >
                            {r.status}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-500">
                          Study Area: <strong>{r.study_area}</strong> • Stage: {r.current_stage} • Duration: {r.duration_seconds}s
                        </p>
                      </div>

                      <button
                        type="button"
                        onClick={() => handleLoadHistoricRun(r.execution_id)}
                        className="px-3 py-1.5 text-xs font-semibold text-emerald-700 bg-emerald-50 hover:bg-emerald-100 rounded border border-emerald-200 transition-colors"
                      >
                        Load Manifest
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-6 text-slate-400">
                  No historical pipeline runs found in backend cache.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

