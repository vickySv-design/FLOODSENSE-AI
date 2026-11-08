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
  Building2,
  Database,
  Globe,
  Info,
  Layers,
  Map as MapIcon,
  Navigation,
  Radio,
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

export default function MapViewport({ analysisData, layerVisibility }) {
  // Default to Real ESRI World Imagery Satellite Photo View as required in Phase 4
  const [baseTileMode, setBaseTileMode] = useState("satellite");

  if (!analysisData) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-950 text-slate-400 font-mono text-sm">
        Awaiting Multi-Modal Satellite Telemetry...
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

  const scenarioKey = `${scenario?.id}_${scenario?.name}_${bbox.join("_")}`;

  const tileConfigs = {
    satellite: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution:
        "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, Copernicus Sentinel-1/2",
    },
    dark: {
      url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      attribution: '&copy; <a href="https://carto.com/">CARTO</a> | Copernicus Sentinel-1/2',
    },
    osm: {
      url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
  };

  const activeTile = tileConfigs[baseTileMode] || tileConfigs.satellite;

  // Real OSM Roads & Buildings
  const allRoadsData = roads_geojson || geojson_layers?.all_roads;
  const blockedRoadsData = blocked_roads_geojson || geojson_layers?.severed_roads;
  const rescueRoutesData = rescue_routes_geojson || geojson_layers?.safe_routes;
  const osmBuildingsData = buildings_geojson || geojson_layers?.buildings;

  const roadSource = data_provenance?.road_data_source || "OpenStreetMap contributors (ODbL)";
  const bldSource = data_provenance?.building_data_source || "OpenStreetMap contributors (ODbL)";
  const isCached = data_provenance?.osm_cache_used ?? true;

  return (
    <div className="relative w-full h-full bg-slate-950 overflow-hidden">
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
            color: "#38bdf8",
            weight: 1.5,
            dashArray: "6, 6",
            fillOpacity: 0.01,
          }}
        />

        {/* Layer 0: Actual OpenStreetMap Road Network */}
        {layerVisibility?.allRoads && allRoadsData?.features?.length > 0 && (
          <GeoJSON
            key={`all_roads_${scenarioKey}`}
            data={allRoadsData}
            style={(feature) => {
              const haz = feature?.properties?.hazard_category;
              const color =
                haz === "POTENTIAL_HAZARD"
                  ? "#ef4444"
                  : haz === "CAUTION"
                  ? "#f59e0b"
                  : haz === "NO_DETECTED_INTERSECTION"
                  ? "#10b981"
                  : "#94a3b8";
              return {
                color: color,
                weight: haz === "POTENTIAL_HAZARD" ? 3.0 : (haz === "CAUTION" ? 2.2 : 1.2),
                opacity: haz ? 0.85 : 0.55,
              };
            }}
            onEachFeature={(feature, layer) => {
              const p = feature.properties || {};
              layer.bindPopup(
                `<div class="font-mono text-xs space-y-1">
                  <div class="font-bold text-slate-100">${p.name || p.road_name || "OSM Street"}</div>
                  <div>Highway Class: <span class="text-cyan-300 font-semibold">${p.highway || "residential"}</span></div>
                  <div>OSM Way ID: <span class="text-slate-400">#${p.osm_id || ""}</span></div>
                  ${p.hazard_category ? `<div>Hazard Grade: <span class="font-bold ${p.hazard_category === 'POTENTIAL_HAZARD' ? 'text-red-400' : p.hazard_category === 'CAUTION' ? 'text-amber-400' : 'text-emerald-400'}">${p.hazard_category}</span></div>` : ""}
                  ${p.flood_intersection_pct !== undefined ? `<div>Flood Overlap: <span class="text-amber-300 font-bold">${p.flood_intersection_pct}%</span></div>` : ""}
                  <div>Bridge Status: <span class="text-amber-300">${p.is_bridge ? "Elevated Bridge Deck" : "Standard Surface"}</span></div>
                  <div>One-Way: ${p.is_oneway ? "Yes" : "Two-Way"} | Surface: ${p.surface || "paved"}</div>
                  <div class="pt-1 border-t border-slate-700 text-[10px] text-emerald-400">
                    &copy; OpenStreetMap contributors (ODbL)
                  </div>
                </div>`
              );
            }}
          />
        )}

        {/* Layer 1: 100m High-Risk Buffer Polygons */}
        {layerVisibility?.highRiskBuffer &&
          geojson_layers?.high_risk_buffer?.features?.length > 0 && (
            <GeoJSON
              key={`buf_${scenarioKey}`}
              data={geojson_layers.high_risk_buffer}
              style={() => ({
                color: "#f59e0b",
                weight: 1.2,
                fillColor: "#f59e0b",
                fillOpacity: 0.24,
                dashArray: "4, 3",
              })}
              onEachFeature={(feature, layer) => {
                layer.bindPopup(
                  `<div class="font-mono text-xs space-y-0.5">
                    <div class="font-bold text-amber-400">100m High-Risk Morphological Buffer</div>
                    <div>Area: <span class="text-slate-100">${feature.properties.area_km2} km²</span></div>
                    <div>Pixels (10m GSD): ${feature.properties.area_pixels}</div>
                  </div>`
                );
              }}
            />
          )}

        {/* Layer 2: Permanent Water Polygons */}
        {layerVisibility?.permanentWater &&
          geojson_layers?.permanent_water?.features?.length > 0 && (
            <GeoJSON
              key={`perm_${scenarioKey}`}
              data={geojson_layers.permanent_water}
              style={() => ({
                color: "#06b6d4",
                weight: 1.6,
                fillColor: "#06b6d4",
                fillOpacity: 0.50,
              })}
              onEachFeature={(feature, layer) => {
                layer.bindPopup(
                  `<div class="font-mono text-xs space-y-0.5">
                    <div class="font-bold text-cyan-400">Permanent Water Body (Class 1)</div>
                    <div>Pre-Event NDWI Baseline: <span class="text-slate-100">${feature.properties.area_km2} km²</span></div>
                  </div>`
                );
              }}
            />
          )}

        {/* Layer 3: Active New Flood Inundation Polygons */}
        {layerVisibility?.newFlood &&
          geojson_layers?.new_flood?.features?.length > 0 && (
            <GeoJSON
              key={`flood_${scenarioKey}`}
              data={geojson_layers.new_flood}
              style={() => ({
                color: "#f43f5e",
                weight: 2,
                fillColor: "#f43f5e",
                fillOpacity: 0.56,
              })}
              onEachFeature={(feature, layer) => {
                layer.bindPopup(
                  `<div class="font-mono text-xs space-y-0.5">
                    <div class="font-bold text-rose-400">Active Flood Inundation (Class 2)</div>
                    <div>Inundated Area: <span class="text-slate-100">${feature.properties.area_km2} km²</span></div>
                    <div>SAR Specular Drop: <span class="text-cyan-300">&lt; -18.5 dB</span></div>
                    <div>Pixels (10m GSD): ${feature.properties.area_pixels}</div>
                  </div>`
                );
              }}
            />
          )}

        {/* Layer 4: Real Blocked Road Segments (Intersecting Flood Mask) */}
        {layerVisibility?.severedRoads &&
          blockedRoadsData?.features?.length > 0 && (
            <GeoJSON
              key={`sev_${scenarioKey}`}
              data={blockedRoadsData}
              style={() => ({
                color: "#ef4444",
                weight: 4,
                opacity: 0.95,
              })}
              onEachFeature={(feature, layer) => {
                const p = feature.properties || {};
                layer.bindPopup(
                  `<div class="font-mono text-xs space-y-1">
                    <div class="font-bold text-red-400 flex items-center gap-1">
                      <span class="w-2 h-2 rounded-full bg-red-500 animate-ping"></span>
                      BLOCKED ROAD CORRIDOR (COLLISION VERIFIED)
                    </div>
                    <div>Corridor: <span class="text-slate-100 font-semibold">${p.road_name || "OSM Street"}</span></div>
                    <div>OSM Way ID: <span class="text-slate-400">#${p.osm_id || ""}</span></div>
                    <div>Highway Class: <span class="text-cyan-300">${p.highway || "road"}</span></div>
                    <div>Reason: <span class="text-rose-400 font-bold">${p.reason || "BLOCKED_ACTIVE_FLOOD"}</span></div>
                    <div>Flood Submergence: <strong class="text-amber-300">${p.submergence_pct ?? 100}%</strong></div>
                    <div>Segment Length: ${p.length_km} km</div>
                    <div class="pt-1 border-t border-slate-700 text-[10px] text-slate-400">
                      Excluded from NetworkX A* navigable routing graph
                    </div>
                  </div>`
                );
              }}
            />
          )}

        {/* Layer 5: Active Safe A* Rescue Routes (Following Real OSM Geometry) */}
        {layerVisibility?.safeRoutes &&
          rescueRoutesData?.features?.length > 0 && (
            <GeoJSON
              key={`routes_${scenarioKey}`}
              data={rescueRoutesData}
              style={(feature) => ({
                color: feature?.properties?.color || "#10b981",
                weight: 5,
                opacity: 0.98,
                className: "rescue-route-animated",
              })}
              onEachFeature={(feature, layer) => {
                const p = feature.properties || {};
                layer.bindPopup(
                  `<div class="font-mono text-xs space-y-1">
                    <div class="font-bold text-emerald-400">${p.route_id} // VERIFIED A* CORRIDOR</div>
                    <div>Target Sector: <span class="text-slate-100 font-semibold">${p.destination}</span></div>
                    <div>OSM Road Distance: <span class="text-cyan-300 font-bold">${p.distance_km} km</span></div>
                    <div>Est. Transit ETA: <span class="text-amber-300 font-bold">${p.eta_minutes} mins</span></div>
                    <div>OSM Way Segments: ${p.segments_count ?? 8} segments</div>
                    <div>Status: <span class="text-emerald-300 font-semibold">${p.status}</span></div>
                    <div class="pt-1 border-t border-slate-700 text-[10px] text-slate-400">
                      Geometry follows verified real-world OpenStreetMap street layout
                    </div>
                  </div>`
                );
              }}
            />
          )}

        {/* Layer 6: Real OpenStreetMap Building Footprints */}
        {layerVisibility?.buildings && osmBuildingsData?.features?.length > 0 ? (
          <GeoJSON
            key={`osm_blds_${scenarioKey}`}
            data={osmBuildingsData}
            style={(feature) => {
              const cat = feature?.properties?.exposure_category || feature?.properties?.damage_status;
              const color =
                cat === "HIGH_EXPOSURE" || cat === "CRITICAL_SUBMERGED"
                  ? "#ef4444"
                  : cat === "MODERATE_EXPOSURE" || cat === "MODERATE_BUFFER_RISK"
                  ? "#f59e0b"
                  : cat === "LOW_OR_NO_DETECTED_EXPOSURE" || cat === "SAFE_DRY"
                  ? "#10b981"
                  : "#94a3b8";
              return {
                color: "#0f172a",
                weight: 0.8,
                fillColor: color,
                fillOpacity: 0.78,
              };
            }}
            onEachFeature={(feature, layer) => {
              const p = feature.properties || {};
              const cat = p.exposure_category || p.damage_status;
              layer.bindPopup(
                `<div class="font-mono text-xs space-y-1">
                  <div class="font-bold text-slate-100">${p.building_name || p.name || "Structure"}</div>
                  <div>Building Type: <span class="text-cyan-300">${p.building_type || "building"}</span></div>
                  <div>OSM ID: <span class="text-slate-400">#${p.building_id || p.osm_id || ""}</span></div>
                  <div>Exposure Category: <span class="font-bold ${cat?.includes('HIGH') ? 'text-red-400' : cat?.includes('MODERATE') ? 'text-amber-400' : 'text-emerald-400'}">${cat}</span></div>
                  ${p.exposed_area_pct !== undefined ? `<div>Footprint Overlap: <span class="text-cyan-300 font-bold">${p.exposed_area_pct}%</span></div>` : ""}
                  ${p.valid_sampled_pixels !== undefined ? `<div class="text-[11px] text-slate-400">Sampled Pixels: ${p.valid_sampled_pixels}</div>` : ""}
                  <div class="text-[10px] text-slate-400 pt-0.5 border-t border-slate-800">
                    Data Source: <span class="text-emerald-400">${p.data_source || "OpenStreetMap"}</span>
                  </div>
                </div>`
              );
            }}
          />
        ) : (
          layerVisibility?.buildings &&
          buildings?.map((bld) => (
            <CircleMarker
              key={bld.id}
              center={[bld.lat, bld.lon]}
              radius={bld.status === "CRITICAL_SUBMERGED" ? 6 : 4.5}
              pathOptions={{
                color: "#020617",
                weight: 1.3,
                fillColor: bld.color,
                fillOpacity: 0.94,
              }}
            >
              <Popup>
                <div className="font-mono text-xs space-y-1">
                  <div className="font-bold text-slate-100">{bld.name}</div>
                  <div>Grade: <span style={{ color: bld.color }} className="font-semibold">{bld.status}</span></div>
                  <div>Source: <span className="text-emerald-400">{bld.data_source || "OSM Real"}</span></div>
                </div>
              </Popup>
            </CircleMarker>
          ))
        )}

        {/* Tactical Command Base & Evacuation Target Markers */}
        {layerVisibility?.safeRoutes &&
          rescue?.waypoints?.map((wp) => {
            const isBase = wp.role === "EMERGENCY_BASE";
            return (
              <CircleMarker
                key={wp.id}
                center={[wp.lat, wp.lon]}
                radius={isBase ? 9 : 8}
                pathOptions={{
                  color: "#ffffff",
                  weight: 2,
                  fillColor: isBase ? "#06b6d4" : "#10b981",
                  fillOpacity: 1,
                }}
              >
                <Tooltip
                  direction="top"
                  offset={[0, -8]}
                  permanent
                  className="!bg-slate-900/95 !text-slate-100 !border !border-cyan-500/50 !font-mono !text-[10px] !px-1.5 !py-0.5 !rounded"
                >
                  {isBase ? "CMD BASE" : wp.id}
                </Tooltip>
                <Popup>
                  <div className="font-mono text-xs space-y-0.5">
                    <div className="font-bold text-cyan-300">{wp.name}</div>
                    <div>Role: {wp.role}</div>
                    <div>Location: [{wp.lat.toFixed(4)}°N, {wp.lon.toFixed(4)}°E]</div>
                    <div className="text-[10px] text-slate-400">Snapped to Nearest OSM Road Node</div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
      </MapContainer>

      {/* Floating Top-Right 3-Mode Basemap Switcher Pill */}
      <div className="absolute top-3 right-3 z-20 flex items-center gap-1 bg-slate-900/95 border border-slate-700/90 rounded-lg p-1 backdrop-blur-md shadow-xl">
        <button
          type="button"
          onClick={() => setBaseTileMode("satellite")}
          className={`inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs font-mono transition ${
            baseTileMode === "satellite"
              ? "bg-cyan-500/25 text-cyan-200 border border-cyan-400/50 font-semibold"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Globe className="w-3 h-3 text-cyan-400" />
          Satellite Imagery (ESRI)
        </button>
        <button
          type="button"
          onClick={() => setBaseTileMode("dark")}
          className={`inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs font-mono transition ${
            baseTileMode === "dark"
              ? "bg-cyan-500/25 text-cyan-200 border border-cyan-400/50 font-semibold"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Layers className="w-3 h-3 text-emerald-400" />
          Dark Tactical GIS
        </button>
        <button
          type="button"
          onClick={() => setBaseTileMode("osm")}
          className={`inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs font-mono transition ${
            baseTileMode === "osm"
              ? "bg-cyan-500/25 text-cyan-200 border border-cyan-400/50 font-semibold"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <MapIcon className="w-3 h-3 text-amber-400" />
          OSM Street Map
        </button>
      </div>

      {/* Floating Top-Left Geographic Provenance & Accuracy Badge */}
      <div className="absolute top-3 left-3 z-20 bg-slate-900/95 border border-slate-800 rounded-lg px-3 py-2 backdrop-blur-md shadow-xl flex items-center gap-2 text-xs font-mono">
        <Database className="w-4 h-4 text-emerald-400 shrink-0" />
        <div>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-200 font-semibold">GEOREFERENCED OSM ROADS</span>
            <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
              {isCached ? "LOCAL CACHE" : "LIVE OSM"}
            </span>
          </div>
          <div className="text-[10px] text-slate-400">
            Roads: <strong className="text-cyan-300">{data_provenance?.road_features_count ?? allRoadsData?.features?.length ?? "--"}</strong> |
            Footprints: <strong className="text-amber-300">{data_provenance?.building_features_count ?? osmBuildingsData?.features?.length ?? "--"}</strong>
          </div>
        </div>
      </div>

      {/* Floating Bottom-Left Tactical Map Legend */}
      <div className="absolute bottom-3 left-3 z-20 bg-slate-900/95 border border-slate-800 rounded-lg p-3 backdrop-blur-md shadow-xl max-w-xs">
        <div className="text-[10px] font-mono uppercase tracking-wider text-cyan-400 font-bold mb-2 flex items-center justify-between">
          <span className="flex items-center gap-1">
            <Radio className="w-3 h-3" />
            Georeferenced GIS Legend
          </span>
          <span className="text-[9px] text-slate-500">EPSG:4326</span>
        </div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-0.5 bg-slate-400" />
            <span className="text-slate-300">OSM Roads</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-1 bg-red-500" />
            <span className="text-slate-200 font-semibold">Blocked Road</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-1 bg-emerald-400 border-b border-dashed border-emerald-300" />
            <span className="text-slate-200 font-semibold">A* Safe Route</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-rose-500/80 border border-rose-400" />
            <span className="text-slate-200">Flood Inundation</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-cyan-400/80 border border-cyan-300" />
            <span className="text-slate-200">Permanent Water</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-amber-400/60 border border-amber-300" />
            <span className="text-slate-200">100m Risk Buffer</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-rose-500" />
            <span className="text-slate-200">Critical Bldg</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
            <span className="text-slate-200">Safe Bldg</span>
          </div>
        </div>
        <div className="mt-2 pt-1.5 border-t border-slate-800 text-[10px] font-mono text-slate-400 flex items-center justify-between">
          <span>&copy; OpenStreetMap contributors</span>
          <span>Esri World Imagery</span>
        </div>
      </div>

      {/* Floating Bottom-Right Active A* Route Telemetry Pill */}
      <div className="hidden sm:flex absolute bottom-3 right-3 z-20 bg-slate-900/95 border border-slate-800 rounded-lg px-3 py-2 backdrop-blur-md items-center gap-3 text-xs font-mono">
        <div className="flex items-center gap-1.5 text-emerald-400">
          <Navigation className="w-3.5 h-3.5" />
          <span>{rescue?.summary?.active_safe_routes || 0} Safe Corridors</span>
        </div>
        <span className="text-slate-700">|</span>
        <div className="flex items-center gap-1.5 text-rose-400">
          <ShieldAlert className="w-3.5 h-3.5" />
          <span>{rescue?.summary?.blocked_road_segments || 0} Blocked Roads</span>
        </div>
      </div>
    </div>
  );
}
