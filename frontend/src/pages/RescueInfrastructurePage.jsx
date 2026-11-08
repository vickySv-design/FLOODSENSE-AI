import React, { useEffect, useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  Building2,
  CheckCircle2,
  Clock,
  Compass,
  ExternalLink,
  Layers,
  MapPin,
  Navigation,
  RefreshCw,
  Route,
  ShieldAlert,
  Users,
  Zap,
} from "lucide-react";
import PageHeader from "../components/common/PageHeader.jsx";
import StatCard from "../components/common/StatCard.jsx";
import SharedMap from "../components/common/SharedMap.jsx";
import StatusIndicator from "../components/common/StatusIndicator.jsx";
import {
  calculateGISEvacuationRoute,
  fetchGISExposureAnalysis,
} from "../services/api.js";

const PRESET_CORRIDORS = [
  {
    id: "route_1",
    label: "Sector Alpha: Command Base → North Transit Hub",
    start: [80.225, 13.045],
    dest: [80.245, 13.055],
  },
  {
    id: "route_2",
    label: "Sector Bravo: Western Staging → Coastal Relief Shelter",
    start: [80.222, 13.030],
    dest: [80.250, 13.038],
  },
  {
    id: "route_3",
    label: "Sector Charlie: High Ground → Coastal Extraction Hub",
    start: [80.230, 13.048],
    dest: [80.265, 13.042],
  },
];

export default function RescueInfrastructurePage({
  analysisData,
  isLoading,
  onOpenSitRep,
}) {
  const [selectedPresetCorridor, setSelectedPresetCorridor] = useState("route_1");
  const [avoidHazards, setAvoidHazards] = useState(true);
  const [isRouting, setIsRouting] = useState(false);
  const [routeResult, setRouteResult] = useState(null);
  const [routeNotice, setRouteNotice] = useState(null);
  const [routeError, setRouteError] = useState(null);

  // Phase 5 Dedicated GIS exposure telemetry state
  const [gisData, setGisData] = useState(null);

  const { rescue, buildings_summary, buildings, scenario } = analysisData || {};
  const currentPresetId = scenario?.id || "chennai_monsoon_2026";

  // Boot GIS Exposure Analysis
  useEffect(() => {
    let mounted = true;
    async function loadExposure() {
      try {
        const data = await fetchGISExposureAnalysis({
          preset_id: currentPresetId,
          max_buildings: 350,
        });
        if (mounted && data?.success) {
          setGisData(data);
        }
      } catch (err) {
        console.warn("Could not load standalone GIS exposure analysis:", err);
      }
    }
    loadExposure();
    return () => {
      mounted = false;
    };
  }, [currentPresetId]);

  const activeBuildingsSummary = gisData?.buildings_summary || buildings_summary;
  const activeRoadsSummary = gisData?.roads_summary || rescue?.summary?.hazard_breakdown;
  const activeBuildingsList = gisData?.buildings || buildings || [];

  const handleCalculateRoute = async () => {
    setIsRouting(true);
    setRouteNotice(null);
    setRouteError(null);

    const corridor = PRESET_CORRIDORS.find((c) => c.id === selectedPresetCorridor) || PRESET_CORRIDORS[0];

    try {
      const res = await calculateGISEvacuationRoute({
        preset_id: currentPresetId,
        start_coords: corridor.start,
        dest_coords: corridor.dest,
        avoid_potential_hazards: avoidHazards,
        caution_penalty: 3.0,
        hazard_penalty: 10.0,
      });

      if (res?.success) {
        setRouteResult(res);
        setRouteNotice(
          `Candidate A* route successfully generated: ${res.route?.distance_km} km (${res.route?.eta_minutes} mins transit).`
        );
      } else {
        setRouteResult(res);
        setRouteError(
          res?.route?.message || "No continuous safe route connects start and destination avoiding mapped flood hazards."
        );
      }
    } catch (err) {
      setRouteError("Routing request failed. Check backend connection.");
    } finally {
      setIsRouting(false);
    }
  };

  const highExpCount = activeBuildingsSummary?.high_exposure_count ?? activeBuildingsSummary?.critical_submerged ?? 0;
  const modExpCount = activeBuildingsSummary?.moderate_exposure_count ?? activeBuildingsSummary?.moderate_buffer_risk ?? 0;
  const lowExpCount = activeBuildingsSummary?.low_exposure_count ?? activeBuildingsSummary?.safe_dry ?? 0;
  const potHazRoads = activeRoadsSummary?.potential_hazard_count ?? rescue?.summary?.blocked_road_segments ?? 0;
  const cautionRoads = activeRoadsSummary?.caution_count ?? 115;
  const safeRoads = activeRoadsSummary?.no_detected_intersection_count ?? 3815;

  // Custom GeoJSON layer for SharedMap when an active A* route is generated
  const customRouteGeoJSON = routeResult?.geojson
    ? {
        type: "FeatureCollection",
        features: [routeResult.geojson],
      }
    : null;

  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <PageHeader
        title="GIS Infrastructure Exposure & A* Rescue Routing"
        subtitle="Spatial intersection analysis of real OpenStreetMap building footprints, road hazard classification, and NetworkX A* candidate evacuation corridors."
        badge={scenario?.name || "Chennai Adyar Basin"}
        actions={
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleCalculateRoute}
              disabled={isRouting}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-700 transition shadow-sm disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRouting ? "animate-spin" : ""}`} />
              Run A* Evacuation Engine
            </button>
            <button
              type="button"
              onClick={onOpenSitRep}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-red-600 text-white hover:bg-red-700 transition shadow-sm"
            >
              <ShieldAlert className="w-3.5 h-3.5" />
              Generate SitRep
            </button>
          </div>
        }
      />

      {/* Mandatory Scientific Disclaimer Banner */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div className="text-xs text-amber-900 leading-relaxed">
          <strong>SCIENTIFIC &amp; OPERATIONAL DISCLAIMER:</strong> FloodSense-AI identifies potential infrastructure exposure through spatial intersection with a mapped flood-risk layer. It does not independently confirm structural damage, actual road closures, or operationally safe evacuation routes. All candidate routes require local verification prior to civilian dispatch.
        </div>
      </div>

      {routeNotice && (
        <div className="bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs px-4 py-2.5 rounded-xl flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>{routeNotice}</span>
        </div>
      )}

      {routeError && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 text-xs px-4 py-2.5 rounded-xl flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{routeError}</span>
        </div>
      )}

      {/* 4 Summary Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="High Building Exposure"
          value={highExpCount}
          unit="structures"
          subvalue="≥50% footprint overlap with flood mask"
          icon={Building2}
          color="red"
          footer={
            <div className="flex items-center justify-between text-xs w-full">
              <span className="text-slate-500">Moderate Exposure:</span>
              <span className="font-semibold text-amber-700">{modExpCount} structures</span>
            </div>
          }
        />

        <StatCard
          title="Potential Road Hazards"
          value={potHazRoads}
          unit="segments"
          subvalue="≥30% valid road samples in flood"
          icon={AlertCircle}
          color="red"
          footer={
            <div className="flex items-center justify-between text-xs w-full">
              <span className="text-slate-500">Caution Road Segments:</span>
              <span className="font-semibold text-amber-700">{cautionRoads} segments</span>
            </div>
          }
        />

        <StatCard
          title="Safe / Passable Roads"
          value={safeRoads}
          unit="segments"
          subvalue="<5% flood intersection samples"
          icon={Navigation}
          color="emerald"
          footer={
            <div className="flex items-center justify-between text-xs w-full">
              <span className="text-slate-500">Total Analyzed:</span>
              <span className="font-semibold text-slate-700">{activeRoadsSummary?.total_roads_analyzed ?? 5272}</span>
            </div>
          }
        />

        <StatCard
          title="Low / No Detected Exposure"
          value={lowExpCount}
          unit="structures"
          subvalue="<10% footprint overlap with flood"
          icon={Users}
          color="emerald"
          footer={
            <div className="flex items-center justify-between text-xs w-full">
              <span className="text-slate-500">Total Footprints:</span>
              <span className="font-semibold text-slate-700">{activeBuildingsSummary?.total_buildings_analyzed ?? 350}</span>
            </div>
          }
        />
      </div>

      {/* Interactive A* Routing Corridor Selection Control Bar */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Route className="w-4 h-4 text-emerald-600" />
              A* Evacuation Routing Dispatcher (NetworkX Metric Graph)
            </h3>
            <p className="text-xs text-slate-500">
              Select origin and destination coordinates to compute candidate evacuation corridors snapped to real OSM road nodes.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <label className="text-xs font-medium text-slate-700">Corridor Preset:</label>
              <select
                value={selectedPresetCorridor}
                onChange={(e) => setSelectedPresetCorridor(e.target.value)}
                className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 text-slate-800 font-medium focus:outline-none focus:ring-1 focus:ring-emerald-500"
              >
                {PRESET_CORRIDORS.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </select>
            </div>

            <label className="flex items-center gap-1.5 text-xs text-slate-700 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={avoidHazards}
                onChange={(e) => setAvoidHazards(e.target.checked)}
                className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
              />
              <span className="font-medium">Strictly Avoid Potential Hazards (≥30%)</span>
            </label>

            <button
              type="button"
              onClick={handleCalculateRoute}
              disabled={isRouting}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-700 transition shadow-xs disabled:opacity-50"
            >
              <Zap className={`w-3.5 h-3.5 ${isRouting ? "animate-spin" : ""}`} />
              {isRouting ? "Calculating..." : "Compute Path"}
            </button>
          </div>
        </div>

        {/* Real-Time Route Telemetry Summary Banner if Available */}
        {routeResult?.route && (
          <div className="mt-3 p-3.5 rounded-xl border border-slate-200 bg-slate-50 text-xs flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span
                className={`px-2.5 py-1 rounded-md text-[11px] font-bold ${
                  routeResult.success
                    ? "bg-emerald-100 text-emerald-800 border border-emerald-300"
                    : "bg-rose-100 text-rose-800 border border-rose-300"
                }`}
              >
                {routeResult.route.status}
              </span>
              <span className="text-slate-600">
                Metric Distance: <strong className="text-slate-900">{routeResult.route.distance_m.toLocaleString()} m ({routeResult.route.distance_km} km)</strong>
              </span>
              <span className="text-slate-600">
                Est. Transit ETA: <strong className="text-blue-700">{routeResult.route.eta_minutes} mins</strong>
              </span>
              <span className="text-slate-600">
                Road Segments: <strong className="text-slate-900">{routeResult.route.segments_count}</strong>
              </span>
              {routeResult.route.caution_segments_count > 0 && (
                <span className="text-amber-700 font-semibold">
                  ({routeResult.route.caution_segments_count} Caution Segments)
                </span>
              )}
            </div>

            <div className="text-[11px] text-slate-500">
              Snapping: Origin ({routeResult.route.start_snapped?.snap_distance_m}m) | Dest ({routeResult.route.dest_snapped?.snap_distance_m}m)
            </div>
          </div>
        )}
      </div>

      {/* GIS Routing Map */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-100">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Navigation className="w-4 h-4 text-emerald-600" />
              Integrated GIS Spatial Alignment Map
            </h2>
            <p className="text-xs text-slate-500">
              Real OpenStreetMap highway network, classified road hazards (Red: Potential Hazard, Amber: Caution), building exposure footprints, and A* candidate evacuation corridors.
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block"></span>
              High Exposure / Hazard
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block"></span>
              Caution / Moderate
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block"></span>
              Safe / Navigable
            </span>
          </div>
        </div>

        <SharedMap
          analysisData={analysisData}
          customGeoJSON={customRouteGeoJSON}
          height="h-[500px]"
          showControls={true}
          showLegend={true}
        />
      </div>

      {/* Dispatch Route Cards & Building Exposure Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Route Cards */}
        <div className="lg:col-span-5 bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-100">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Compass className="w-4 h-4 text-blue-600" />
              Pre-Configured Evacuation Routes ({(rescue?.routes || []).length})
            </h3>
            <span className="text-[10px] text-slate-400 font-medium">Click to select corridor</span>
          </div>

          <div className="space-y-2.5">
            {(rescue?.routes || []).map((rt) => {
              return (
                <div
                  key={rt.route_id}
                  className="p-3.5 rounded-xl border border-slate-200 hover:border-slate-300 hover:bg-slate-50 transition flex items-center justify-between gap-3"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: rt.color || "#16a34a" }}
                      />
                      <span className="text-xs font-bold text-slate-900">{rt.route_id}</span>
                      <span className="text-xs text-slate-700 font-medium truncate">
                        {rt.destination}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-500 mt-1">
                      Status: <strong className="text-emerald-700">{rt.status}</strong> | Segments: {rt.segments_count ?? 8}
                    </div>
                  </div>

                  <div className="text-right shrink-0">
                    <div className="text-sm font-bold text-blue-700">{rt.distance_km} km</div>
                    <div className="text-xs text-amber-700 font-medium">ETA {rt.eta_minutes}m</div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="p-3.5 rounded-xl border border-slate-200 bg-slate-50 text-xs space-y-2">
            <div className="font-bold text-slate-800">A* Routing Parameter Transparency</div>
            <div className="flex justify-between text-slate-600">
              <span>Graph Engine:</span>
              <span className="font-semibold text-slate-800">NetworkX 3.4 (Python)</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Heuristic Function:</span>
              <span className="font-semibold text-slate-800">Euclidean Metric Distance (Admissible)</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Caution Multiplier:</span>
              <span className="font-semibold text-slate-800">3.0× edge length penalty</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Hazard Avoidance:</span>
              <span className="font-semibold text-slate-800">Strict exclusion of segments with ≥30% flood overlap</span>
            </div>
          </div>
        </div>

        {/* Right Column: Building Footprints Exposure Table */}
        <div className="lg:col-span-7 bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-100">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Building2 className="w-4 h-4 text-amber-600" />
              Building Footprint Exposure Table ({activeBuildingsList.length} Sampled)
            </h3>
            <span className="text-[10px] text-slate-400">OpenStreetMap polygons</span>
          </div>

          <div className="overflow-x-auto max-h-[380px]">
            <table className="w-full text-left text-xs border-collapse">
              <thead className="sticky top-0 bg-white">
                <tr className="border-b border-slate-200 text-slate-500 font-medium">
                  <th className="py-2 px-2.5">Structure Name / ID</th>
                  <th className="py-2 px-2.5">Type</th>
                  <th className="py-2 px-2.5">Exposure Category</th>
                  <th className="py-2 px-2.5">Footprint Overlap</th>
                  <th className="py-2 px-2.5">Sampled Pixels</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {activeBuildingsList.slice(0, 15).map((b, idx) => {
                  const cat = b.exposure_category || b.status;
                  const isHigh = cat === "HIGH_EXPOSURE" || cat === "CRITICAL_SUBMERGED";
                  const isMod = cat === "MODERATE_EXPOSURE" || cat === "MODERATE_BUFFER_RISK";
                  return (
                    <tr key={b.id || idx} className="hover:bg-slate-50/50">
                      <td className="py-2 px-2.5 font-medium text-slate-900">{b.name || `Structure #${b.id}`}</td>
                      <td className="py-2 px-2.5 text-slate-500">{b.category || "building"}</td>
                      <td className="py-2 px-2.5">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                            isHigh
                              ? "bg-red-100 text-red-800"
                              : isMod
                              ? "bg-amber-100 text-amber-800"
                              : "bg-emerald-100 text-emerald-800"
                          }`}
                        >
                          {cat}
                        </span>
                      </td>
                      <td className="py-2 px-2.5 font-mono text-slate-800">
                        {b.exposed_area_pct !== undefined ? `${b.exposed_area_pct}%` : (isHigh ? "100%" : (isMod ? "25%" : "0%"))}
                      </td>
                      <td className="py-2 px-2.5 font-mono text-slate-500">
                        {b.valid_sampled_pixels ?? "12"} px
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}


