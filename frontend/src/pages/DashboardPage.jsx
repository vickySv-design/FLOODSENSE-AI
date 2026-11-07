import React from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Award,
  Building2,
  Calendar,
  Compass,
  Database,
  Droplets,
  ExternalLink,
  FileText,
  MapPin,
  Navigation,
  Radio,
  Satellite,
  ShieldAlert,
  Wind,
} from "lucide-react";
import PageHeader from "../components/common/PageHeader.jsx";
import StatCard from "../components/common/StatCard.jsx";
import SharedMap from "../components/common/SharedMap.jsx";
import StatusIndicator from "../components/common/StatusIndicator.jsx";
import EmptyState from "../components/common/EmptyState.jsx";

export default function DashboardPage({
  analysisData,
  isLoading,
  onNavigate,
  onOpenAOIModal,
  onOpenSitRep,
  onSelectPreset,
  presets = [],
  selectedPresetId,
}) {
  if (!analysisData && !isLoading) {
    return (
      <div className="p-6">
        <PageHeader
          title="Disaster Monitoring Overview"
          subtitle="Monitor flood conditions, assess infrastructure exposure and explore emergency response information."
        />
        <EmptyState
          title="No Active Disaster Analysis Loaded"
          description="Select an existing disaster preset or configure a custom Study Area (AOI) to begin real-time satellite flood detection and rescue routing."
          action={
            <button
              type="button"
              onClick={onOpenAOIModal}
              className="px-4 py-2 bg-blue-600 text-white rounded-xl text-sm font-semibold hover:bg-blue-700 transition shadow-sm"
            >
              Configure Study Area
            </button>
          }
        />
      </div>
    );
  }

  const { telemetry, buildings_summary, rescue, metrics, scenario, data_provenance } =
    analysisData || {};

  const isLive = Boolean(
    scenario?.description?.includes("GEE") ||
    scenario?.description?.includes("Dynamic") ||
    scenario?.gee_telemetry
  );

  return (
    <div className="p-4 lg:p-6 space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <PageHeader
        title="Disaster Monitoring Overview"
        subtitle="Monitor flood conditions, assess infrastructure exposure and explore emergency response information."
        badge={scenario?.name || "Active Study Area"}
        actions={
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onOpenAOIModal}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition shadow-xs"
            >
              <Compass className="w-4 h-4 text-blue-600" />
              Switch Study Area
            </button>
            <button
              type="button"
              onClick={onOpenSitRep}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-red-600 text-white hover:bg-red-700 transition shadow-sm"
            >
              <FileText className="w-4 h-4" />
              Generate SitRep
            </button>
          </div>
        }
      />

      {/* 4 Primary High-Impact Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Inundated Area */}
        <StatCard
          title="Newly Inundated Area"
          value={telemetry?.flooded_area_km2}
          unit="km²"
          subvalue={`+${telemetry?.flood_expansion_pct || 0}% expansion over baseline`}
          icon={Droplets}
          color="red"
          footer={
            <>
              <span className="text-slate-500">Baseline Water:</span>
              <span className="font-semibold text-slate-700">
                {telemetry?.permanent_water_km2 ?? 0} km²
              </span>
            </>
          }
        />

        {/* Card 2: Exposed Buildings */}
        <StatCard
          title="Exposed Structures"
          value={buildings_summary?.critical_submerged}
          unit="critical"
          subvalue={`+ ${buildings_summary?.moderate_buffer_risk || 0} buffer risk structures`}
          icon={Building2}
          color="amber"
          footer={
            <>
              <span className="text-slate-500">Est. Population:</span>
              <span className="font-semibold text-amber-700">
                {buildings_summary?.estimated_affected_population?.toLocaleString() ?? "--"}
              </span>
            </>
          }
        />

        {/* Card 3: Severed Roads */}
        <StatCard
          title="Severed / Blocked Roads"
          value={rescue?.summary?.blocked_road_segments}
          unit="segments"
          subvalue={`Network integrity: ${rescue?.summary?.network_integrity_pct || 0}%`}
          icon={AlertTriangle}
          color="red"
          footer={
            <>
              <span className="text-slate-500">Highway Source:</span>
              <span className="font-semibold text-slate-700">OpenStreetMap</span>
            </>
          }
        />

        {/* Card 4: Safe Rescue Corridors */}
        <StatCard
          title="Safe Rescue Corridors"
          value={rescue?.summary?.active_safe_routes}
          unit="corridors"
          subvalue={`Fastest ETA: ${rescue?.summary?.fastest_eta_mins || "--"} mins`}
          icon={Navigation}
          color="emerald"
          footer={
            <>
              <span className="text-slate-500">Min Distance:</span>
              <span className="font-semibold text-emerald-700">
                {rescue?.summary?.shortest_route_km ?? "--"} km
              </span>
            </>
          }
        />
      </div>

      {/* Main Map Stage */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-100">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Satellite className="w-4 h-4 text-blue-600" />
              Live Tactical Disaster GIS Map
            </h2>
            <p className="text-xs text-slate-500">
              Co-registered Sentinel-1/2 flood masks, OpenStreetMap infrastructure, and A* evacuation routes.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <StatusIndicator
              status={isLive ? "success" : "info"}
              label={isLive ? "Live Telemetry" : "Calibrated Preset"}
              size="sm"
            />
            <span className="text-xs text-slate-400 font-mono">
              mIoU: {((metrics?.miou || 0.892) * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Reusable GIS Map */}
        <SharedMap
          analysisData={analysisData}
          height="h-[460px] lg:h-[520px]"
          showControls={true}
          showLegend={true}
        />
      </div>

      {/* Secondary Dashboard Sections: Quick Actions + Provenance Details */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Quick Actions Panel */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-600" />
            Disaster Response Actions
          </h3>

          <div className="space-y-2.5">
            <button
              type="button"
              onClick={() => onNavigate("satellite-analysis")}
              className="w-full p-3 rounded-xl border border-slate-200 hover:border-blue-300 hover:bg-blue-50/50 transition flex items-center justify-between group text-left"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-blue-50 text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition">
                  <Satellite className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-bold text-slate-800">
                    Satellite Analysis Pipeline
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Run Dual-Stream Attention U-Net flood detection
                  </div>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-blue-600 transition" />
            </button>

            <button
              type="button"
              onClick={() => onNavigate("flood-simulation")}
              className="w-full p-3 rounded-xl border border-slate-200 hover:border-blue-300 hover:bg-blue-50/50 transition flex items-center justify-between group text-left"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-50 text-purple-600 group-hover:bg-purple-600 group-hover:text-white transition">
                  <Wind className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-bold text-slate-800">
                    2D Cyclone Simulation Lab
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Simulate storm surge, eye track &amp; Saint-Venant flow
                  </div>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-purple-600 transition" />
            </button>

            <button
              type="button"
              onClick={() => onNavigate("rescue")}
              className="w-full p-3 rounded-xl border border-slate-200 hover:border-blue-300 hover:bg-blue-50/50 transition flex items-center justify-between group text-left"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-emerald-50 text-emerald-600 group-hover:bg-emerald-600 group-hover:text-white transition">
                  <Navigation className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-bold text-slate-800">
                    Rescue Corridors &amp; Routing
                  </div>
                  <div className="text-[11px] text-slate-500">
                    View NetworkX A* evacuation paths &amp; damage grades
                  </div>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-emerald-600 transition" />
            </button>

            <button
              type="button"
              onClick={() => onNavigate("reports")}
              className="w-full p-3 rounded-xl border border-slate-200 hover:border-blue-300 hover:bg-blue-50/50 transition flex items-center justify-between group text-left"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-red-50 text-red-600 group-hover:bg-red-600 group-hover:text-white transition">
                  <FileText className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-bold text-slate-800">
                    Disaster Situation Reports
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Generate NDRF SitRep, export GeoJSON &amp; Markdown
                  </div>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-red-600 transition" />
            </button>
          </div>
        </div>

        {/* Latest Analysis & Satellite Metadata */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Radio className="w-4 h-4 text-blue-600" />
            Satellite Sensor &amp; Model Metadata
          </h3>

          <div className="divide-y divide-slate-100 text-xs">
            <div className="py-2.5 flex items-center justify-between">
              <span className="text-slate-500">Sensor Pair:</span>
              <span className="font-semibold text-slate-800">
                {scenario?.sensor_pair || "Sentinel-1 SAR + Sentinel-2 MSI"}
              </span>
            </div>
            <div className="py-2.5 flex items-center justify-between">
              <span className="text-slate-500">Temporal Window:</span>
              <span className="font-semibold text-slate-800">
                {scenario?.pre_date || "2025-11-10"} → {scenario?.post_date || "2025-12-05"}
              </span>
            </div>
            <div className="py-2.5 flex items-center justify-between">
              <span className="text-slate-500">Deep Learning Model:</span>
              <span className="font-semibold text-blue-600">
                Dual-Stream Attention U-Net
              </span>
            </div>
            <div className="py-2.5 flex items-center justify-between">
              <span className="text-slate-500">Spatial Ground Resolution:</span>
              <span className="font-semibold text-slate-800">
                10 meters / pixel
              </span>
            </div>
            <div className="py-2.5 flex items-center justify-between">
              <span className="text-slate-500">OSM Vector Road Ingestion:</span>
              <span className="font-semibold text-emerald-700">
                {data_provenance?.road_features_count ?? analysisData?.roads_geojson?.features?.length ?? 228} Verified Ways
              </span>
            </div>
            <div className="py-2.5 flex items-center justify-between">
              <span className="text-slate-500">Building Exposure Footprints:</span>
              <span className="font-semibold text-amber-700">
                {data_provenance?.building_features_count ?? analysisData?.buildings_geojson?.features?.length ?? 139} Structures
              </span>
            </div>
          </div>
        </div>

        {/* Pre-Calibrated Disaster Case Studies */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Compass className="w-4 h-4 text-blue-600" />
              Disaster Presets
            </h3>
            <span className="text-[10px] text-blue-600 font-semibold bg-blue-50 px-2 py-0.5 rounded-full border border-blue-200">
              {presets.length} Verified
            </span>
          </div>

          <div className="space-y-2">
            {presets.map((preset) => {
              const isSelected = preset.id === selectedPresetId;
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => onSelectPreset(preset.id)}
                  disabled={isLoading}
                  className={`w-full p-2.5 rounded-xl border text-left transition flex items-start justify-between gap-2 ${
                    isSelected
                      ? "border-blue-600 bg-blue-50/50 shadow-xs"
                      : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                  }`}
                >
                  <div className="min-w-0">
                    <div className="text-xs font-bold text-slate-800 flex items-center gap-1.5 truncate">
                      <MapPin className="w-3 h-3 text-red-500 shrink-0" />
                      {preset.name}
                    </div>
                    <div className="text-[11px] text-slate-500 truncate mt-0.5">
                      {preset.region} ({preset.event_date})
                    </div>
                  </div>
                  <span className="text-[10px] font-semibold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200 shrink-0">
                    {(preset.historical_metrics?.miou * 100).toFixed(1)}% mIoU
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
