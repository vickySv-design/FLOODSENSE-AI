import React, { useEffect, useState } from "react";
import {
  CircleMarker,
  GeoJSON,
  MapContainer,
  Popup,
  Rectangle,
  TileLayer,
  Tooltip,
  useMap,
} from "react-leaflet";
import {
  Globe,
  Layers,
  Map as MapIcon,
  Maximize2,
  Minimize2,
  ShieldAlert,
} from "lucide-react";

function MapBoundsController({ bbox }) {
  const map = useMap();
  useEffect(() => {
    if (bbox && bbox.length === 4) {
      const [minLat, minLon, maxLat, maxLon] = bbox;
      map.fitBounds(
        [
          [minLat, minLon],
          [maxLat, maxLon],
        ],
        { padding: [24, 24], animate: true, duration: 0.9 }
      );
    }
  }, [bbox, map]);
  return null;
}

export default function SharedMap({
  analysisData,
  customGeoJSON,
  layerVisibility: initialVisibility,
  height = "h-[540px]",
  showControls = true,
  showLegend = true,
}) {
  const [baseTileMode, setBaseTileMode] = useState("satellite"); // "satellite" | "osm" | "carto_light"
  const [layersOpen, setLayersOpen] = useState(false);
  const [legendOpen, setLegendOpen] = useState(true);

  const [layers, setLayers] = useState({
    allRoads: true,
    newFlood: true,
    permanentWater: true,
    highRiskBuffer: true,
    buildings: true,
    safeRoutes: true,
    severedRoads: true,
    ...initialVisibility,
  });

  const toggleLayer = (k) => setLayers((prev) => ({ ...prev, [k]: !prev[k] }));

  if (!analysisData) {
    return (
      <div className={`w-full ${height} rounded-xl border border-slate-200 bg-slate-50 flex flex-col items-center justify-center p-6 text-center text-slate-500`}>
        <Globe className="w-10 h-10 text-slate-300 mb-2" />
        <p className="text-sm font-medium text-slate-700">Awaiting Spatial & Geospatial Telemetry</p>
        <p className="text-xs text-slate-400 mt-1 max-w-sm">
          Select a Study Area or run a Satellite Analysis to load real OpenStreetMap roads, buildings, and flood inundation vectors.
        </p>
      </div>
    );
  }

  const {
    scenario,
    geojson_layers,
    roads_geojson,
    blocked_roads_geojson,
    rescue_routes_geojson,
    buildings_geojson,
    buildings,
    rescue,
    data_provenance,
  } = analysisData;

  const bbox = scenario?.bbox || [13.01, 80.22, 13.06, 80.27];
  const center = scenario?.center || [13.035, 80.245];
  const bounds = [
    [bbox[0], bbox[1]],
    [bbox[2], bbox[3]],
  ];

  const scenarioKey = `${scenario?.id || "scen"}_${bbox.join("_")}`;

  const tileConfigs = {
    satellite: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution: "Tiles &copy; Esri, Maxar, Earthstar Geographics",
    },
    osm: {
      url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      attribution: "&copy; OpenStreetMap contributors",
    },
    carto_light: {
      url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
      attribution: "&copy; CARTO, OpenStreetMap",
    },
  };

  const activeTile = tileConfigs[baseTileMode] || tileConfigs.satellite;

  const allRoadsData = roads_geojson || geojson_layers?.all_roads;
  const blockedRoadsData = blocked_roads_geojson || geojson_layers?.severed_roads;
  const rescueRoutesData = rescue_routes_geojson || geojson_layers?.safe_routes;
  const osmBuildingsData = buildings_geojson || geojson_layers?.buildings;

  return (
    <div className={`relative w-full ${height} rounded-xl overflow-hidden border border-slate-200 shadow-sm bg-slate-100`}>
      <MapContainer
        center={center}
        zoom={scenario?.zoom || 14}
        className="w-full h-full z-10"
        zoomControl={true}
      >
        <TileLayer
          key={baseTileMode}
          url={activeTile.url}
          attribution={activeTile.attribution}
        />

        <MapBoundsController bbox={bbox} />

        {/* AOI Bounding Box Frame */}
        <Rectangle
          bounds={bounds}
          pathOptions={{
            color: "#2563eb",
            weight: 2,
            dashArray: "6, 6",
            fillOpacity: 0.02,
          }}
        />

        {/* Optional Custom GeoJSON (e.g. from scenario snapshots) */}
        {customGeoJSON && (
          <GeoJSON
            key={`custom_${scenarioKey}_${JSON.stringify(customGeoJSON).length}`}
            data={customGeoJSON}
            style={(feature) => ({
              color: feature?.properties?.stroke || "#dc2626",
              weight: 2,
              fillColor: feature?.properties?.fill || "#ef4444",
              fillOpacity: feature?.properties?.["fill-opacity"] ?? 0.5,
            })}
          />
        )}

        {/* Layer 0: Actual OpenStreetMap Road Network */}
        {layers.allRoads && allRoadsData?.features?.length > 0 && (
          <GeoJSON
            key={`all_roads_${scenarioKey}`}
            data={allRoadsData}
            style={(feature) => ({
              color: "#64748b",
              weight: feature?.properties?.highway?.includes("primary") || feature?.properties?.highway?.includes("trunk") ? 2.5 : 1.2,
              opacity: 0.5,
            })}
            onEachFeature={(feature, layer) => {
              const p = feature.properties || {};
              layer.bindPopup(
                `<div class="text-xs space-y-1">
                  <div class="font-bold text-slate-900">${p.name || "OSM Street"}</div>
                  <div class="text-slate-600">Class: <span class="text-blue-600 font-semibold">${p.highway || "residential"}</span></div>
                  <div class="text-slate-500">Way ID: #${p.osm_id || ""}</div>
                  <div>Status: <span class="text-emerald-700 font-medium">${p.is_bridge ? "Bridge Deck" : "Standard Road"}</span></div>
                  <div class="pt-1 border-t border-slate-200 text-[10px] text-slate-400">
                    &copy; OpenStreetMap contributors
                  </div>
                </div>`
              );
            }}
          />
        )}

        {/* Layer 1: 100m High-Risk Buffer */}
        {layers.highRiskBuffer && geojson_layers?.high_risk_buffer?.features?.length > 0 && (
          <GeoJSON
            key={`buf_${scenarioKey}`}
            data={geojson_layers.high_risk_buffer}
            style={() => ({
              color: "#d97706",
              weight: 1.2,
              fillColor: "#f59e0b",
              fillOpacity: 0.22,
              dashArray: "4, 3",
            })}
            onEachFeature={(feature, layer) => {
              layer.bindPopup(
                `<div class="text-xs space-y-0.5">
                  <div class="font-bold text-amber-700">100m Morphological Buffer (High Risk)</div>
                  <div>Buffer Area: <span class="font-semibold text-slate-900">${feature.properties.area_km2} km²</span></div>
                  <div class="text-slate-500">Pixel Count: ${feature.properties.area_pixels}</div>
                </div>`
              );
            }}
          />
        )}

        {/* Layer 2: Permanent Water Baseline */}
        {layers.permanentWater && geojson_layers?.permanent_water?.features?.length > 0 && (
          <GeoJSON
            key={`perm_${scenarioKey}`}
            data={geojson_layers.permanent_water}
            style={() => ({
              color: "#0891b2",
              weight: 1.5,
              fillColor: "#06b6d4",
              fillOpacity: 0.45,
            })}
            onEachFeature={(feature, layer) => {
              layer.bindPopup(
                `<div class="text-xs space-y-0.5">
                  <div class="font-bold text-cyan-700">Permanent Water Body (Baseline)</div>
                  <div>Baseline Area: <span class="font-semibold text-slate-900">${feature.properties.area_km2} km²</span></div>
                </div>`
              );
            }}
          />
        )}

        {/* Layer 3: New Flood Inundation */}
        {layers.newFlood && geojson_layers?.new_flood?.features?.length > 0 && (
          <GeoJSON
            key={`flood_${scenarioKey}`}
            data={geojson_layers.new_flood}
            style={() => ({
              color: "#dc2626",
              weight: 2,
              fillColor: "#ef4444",
              fillOpacity: 0.55,
            })}
            onEachFeature={(feature, layer) => {
              layer.bindPopup(
                `<div class="text-xs space-y-0.5">
                  <div class="font-bold text-red-600">Active Flood Inundation (Class 2)</div>
                  <div>Inundated Area: <span class="font-bold text-slate-900">${feature.properties.area_km2} km²</span></div>
                  <div class="text-slate-500">SAR Specular Drop: &lt; -16 dB</div>
                </div>`
              );
            }}
          />
        )}

        {/* Layer 4: Blocked Road Corridors */}
        {layers.severedRoads && blockedRoadsData?.features?.length > 0 && (
          <GeoJSON
            key={`sev_${scenarioKey}`}
            data={blockedRoadsData}
            style={() => ({
              color: "#dc2626",
              weight: 4,
              opacity: 0.9,
            })}
            onEachFeature={(feature, layer) => {
              const p = feature.properties || {};
              layer.bindPopup(
                `<div class="text-xs space-y-1">
                  <div class="font-bold text-red-600 flex items-center gap-1">
                    <span class="w-2 h-2 rounded-full bg-red-600 inline-block"></span>
                    BLOCKED ROAD CORRIDOR
                  </div>
                  <div>Corridor: <span class="font-semibold text-slate-900">${p.road_name || "Street"}</span></div>
                  <div>Submergence: <span class="font-bold text-amber-700">${p.submergence_pct ?? 100}%</span></div>
                  <div class="text-slate-500">Length: ${p.length_km} km</div>
                  <div class="pt-1 border-t border-slate-200 text-[10px] text-red-600">
                    Excluded from navigable A* routing
                  </div>
                </div>`
              );
            }}
          />
        )}

        {/* Layer 5: Active Safe Rescue Routes */}
        {layers.safeRoutes && rescueRoutesData?.features?.length > 0 && (
          <GeoJSON
            key={`routes_${scenarioKey}`}
            data={rescueRoutesData}
            style={(feature) => ({
              color: feature?.properties?.color || "#16a34a",
              weight: 5,
              opacity: 0.95,
              className: "rescue-route-animated",
            })}
            onEachFeature={(feature, layer) => {
              const p = feature.properties || {};
              layer.bindPopup(
                `<div class="text-xs space-y-1">
                  <div class="font-bold text-emerald-700">${p.route_id} // SAFE CORRIDOR</div>
                  <div>Destination: <span class="font-semibold text-slate-900">${p.destination}</span></div>
                  <div>Distance: <span class="font-bold text-blue-600">${p.distance_km} km</span></div>
                  <div>Est. Transit: <span class="font-bold text-amber-600">${p.eta_minutes} mins</span></div>
                  <div class="pt-1 border-t border-slate-200 text-[10px] text-slate-500">
                    Calculated via NetworkX A* Heuristic
                  </div>
                </div>`
              );
            }}
          />
        )}

        {/* Layer 6: OpenStreetMap Building Footprints */}
        {layers.buildings && osmBuildingsData?.features?.length > 0 ? (
          <GeoJSON
            key={`osm_blds_${scenarioKey}`}
            data={osmBuildingsData}
            style={(feature) => {
              const status = feature?.properties?.damage_status;
              const color =
                status === "CRITICAL_SUBMERGED"
                  ? "#dc2626"
                  : status === "MODERATE_BUFFER_RISK"
                  ? "#d97706"
                  : "#16a34a";
              return {
                color: "#475569",
                weight: 0.6,
                fillColor: color,
                fillOpacity: 0.75,
              };
            }}
            onEachFeature={(feature, layer) => {
              const p = feature.properties || {};
              layer.bindPopup(
                `<div class="text-xs space-y-1">
                  <div class="font-bold text-slate-900">${p.building_name || "Structure"}</div>
                  <div>Type: <span class="text-slate-700">${p.building_type || "building"}</span></div>
                  <div>Damage Grade: <span class="font-bold" style="color: ${p.color}">${p.damage_status}</span></div>
                  <div class="text-slate-600">Est. Depth: <span class="font-semibold text-red-600">${p.estimated_water_depth_ft ?? 0.0} ft</span></div>
                </div>`
              );
            }}
          />
        ) : (
          layers.buildings &&
          buildings?.map((bld) => (
            <CircleMarker
              key={bld.id}
              center={[bld.lat, bld.lon]}
              radius={bld.status === "CRITICAL_SUBMERGED" ? 5 : 4}
              pathOptions={{
                color: "#ffffff",
                weight: 1.2,
                fillColor: bld.status === "CRITICAL_SUBMERGED" ? "#dc2626" : "#16a34a",
                fillOpacity: 0.9,
              }}
            >
              <Popup>
                <div className="text-xs">
                  <div className="font-bold text-slate-900">{bld.name}</div>
                  <div className="text-slate-600">Grade: {bld.status}</div>
                </div>
              </Popup>
            </CircleMarker>
          ))
        )}

        {/* Waypoints & Command Base */}
        {layers.safeRoutes &&
          rescue?.waypoints?.map((wp) => {
            const isBase = wp.role === "EMERGENCY_BASE";
            return (
              <CircleMarker
                key={wp.id}
                center={[wp.lat, wp.lon]}
                radius={isBase ? 8 : 7}
                pathOptions={{
                  color: "#ffffff",
                  weight: 2,
                  fillColor: isBase ? "#2563eb" : "#16a34a",
                  fillOpacity: 1,
                }}
              >
                <Tooltip direction="top" offset={[0, -8]} permanent className="!bg-white !text-slate-800 !border !border-slate-300 !font-sans !font-semibold !text-[11px] !px-2 !py-0.5 !rounded-md !shadow-sm">
                  {isBase ? "HQ Base" : wp.id}
                </Tooltip>
              </CircleMarker>
            );
          })}
      </MapContainer>

      {/* Floating Basemap Selector Pill */}
      {showControls && (
        <div className="absolute top-3 right-3 z-20 flex items-center gap-1 bg-white/95 border border-slate-200 rounded-lg p-1 shadow-md text-xs backdrop-blur-sm">
          <button
            type="button"
            onClick={() => setBaseTileMode("satellite")}
            className={`px-2.5 py-1 rounded font-medium transition ${
              baseTileMode === "satellite"
                ? "bg-blue-50 text-blue-700 font-semibold border border-blue-200"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Satellite
          </button>
          <button
            type="button"
            onClick={() => setBaseTileMode("osm")}
            className={`px-2.5 py-1 rounded font-medium transition ${
              baseTileMode === "osm"
                ? "bg-blue-50 text-blue-700 font-semibold border border-blue-200"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            OpenStreetMap
          </button>
          <button
            type="button"
            onClick={() => setBaseTileMode("carto_light")}
            className={`px-2.5 py-1 rounded font-medium transition ${
              baseTileMode === "carto_light"
                ? "bg-blue-50 text-blue-700 font-semibold border border-blue-200"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Light Canvas
          </button>
        </div>
      )}

      {/* Floating Layer Visibility Drawer Pill */}
      {showControls && (
        <div className="absolute top-3 left-3 z-20">
          <div className="bg-white/95 border border-slate-200 rounded-lg shadow-md backdrop-blur-sm overflow-hidden max-w-xs">
            <button
              type="button"
              onClick={() => setLayersOpen(!layersOpen)}
              className="w-full px-3 py-2 text-xs font-semibold text-slate-800 flex items-center justify-between gap-2 hover:bg-slate-50 transition"
            >
              <span className="flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-blue-600" />
                Map Layers
              </span>
              <span className="text-[10px] text-slate-400">
                {layersOpen ? "Hide" : "Filter"}
              </span>
            </button>

            {layersOpen && (
              <div className="p-3 border-t border-slate-100 space-y-1.5 text-xs bg-white">
                <label className="flex items-center justify-between gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-600" />
                    New Flood (Class 2)
                  </span>
                  <input
                    type="checkbox"
                    checked={layers.newFlood}
                    onChange={() => toggleLayer("newFlood")}
                    className="accent-blue-600 rounded"
                  />
                </label>
                <label className="flex items-center justify-between gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-500" />
                    Permanent Water (Class 1)
                  </span>
                  <input
                    type="checkbox"
                    checked={layers.permanentWater}
                    onChange={() => toggleLayer("permanentWater")}
                    className="accent-blue-600 rounded"
                  />
                </label>
                <label className="flex items-center justify-between gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                    100m High-Risk Buffer
                  </span>
                  <input
                    type="checkbox"
                    checked={layers.highRiskBuffer}
                    onChange={() => toggleLayer("highRiskBuffer")}
                    className="accent-blue-600 rounded"
                  />
                </label>
                <label className="flex items-center justify-between gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-600" />
                    Safe A* Corridors
                  </span>
                  <input
                    type="checkbox"
                    checked={layers.safeRoutes}
                    onChange={() => toggleLayer("safeRoutes")}
                    className="accent-blue-600 rounded"
                  />
                </label>
                <label className="flex items-center justify-between gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
                    Severed / Blocked Roads
                  </span>
                  <input
                    type="checkbox"
                    checked={layers.severedRoads}
                    onChange={() => toggleLayer("severedRoads")}
                    className="accent-blue-600 rounded"
                  />
                </label>
                <label className="flex items-center justify-between gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-slate-400" />
                    OSM Road Network
                  </span>
                  <input
                    type="checkbox"
                    checked={layers.allRoads}
                    onChange={() => toggleLayer("allRoads")}
                    className="accent-blue-600 rounded"
                  />
                </label>
                <label className="flex items-center justify-between gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-indigo-500" />
                    Building Footprints
                  </span>
                  <input
                    type="checkbox"
                    checked={layers.buildings}
                    onChange={() => toggleLayer("buildings")}
                    className="accent-blue-600 rounded"
                  />
                </label>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Floating Interactive Legend */}
      {showLegend && (
        <div className="absolute bottom-3 left-3 z-20">
          <div className="bg-white/95 border border-slate-200 rounded-lg p-2.5 shadow-md backdrop-blur-sm text-xs">
            <div className="font-semibold text-slate-800 mb-1.5 flex items-center justify-between gap-3">
              <span>Map Legend</span>
              <span className="text-[10px] text-slate-400">10m Pixel Grid</span>
            </div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-slate-600">
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-2 rounded-sm bg-red-600 inline-block" />
                <span>Flood (Class 2)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-2 rounded-sm bg-cyan-500 inline-block" />
                <span>Permanent Water</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-2 rounded-sm bg-amber-400 inline-block" />
                <span>100m Risk Buffer</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-1.5 rounded-full bg-emerald-600 inline-block" />
                <span>Safe Corridor</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-1.5 rounded-full bg-red-500 inline-block" />
                <span>Blocked Road</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-2 rounded-sm bg-slate-400 inline-block" />
                <span>OSM Road Net</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
