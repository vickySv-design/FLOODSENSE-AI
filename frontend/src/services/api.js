import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 45000,
});

export async function fetchHealthStatus() {
  const { data } = await apiClient.get("/api/health");
  return data;
}

export async function fetchDisasterPresets() {
  const { data } = await apiClient.get("/api/presets");
  return data;
}

export async function analyzePresetScenario(presetId) {
  const formData = new FormData();
  formData.append("preset_id", presetId);
  const { data } = await apiClient.post("/api/analyze", formData);
  return data;
}

export async function analyzeCustomUpload({ preOpticalFile, postSarFile, customName, bbox }) {
  const formData = new FormData();
  formData.append("pre_optical_file", preOpticalFile);
  formData.append("post_sar_file", postSarFile);
  if (customName) {
    formData.append("custom_name", customName);
  }
  if (bbox) {
    formData.append("bbox_json", JSON.stringify(bbox));
  }
  const { data } = await apiClient.post("/api/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function fetchAndAnalyzeGEETiles(payload) {
  const { data } = await apiClient.post("/api/gee-fetch", payload);
  return data;
}

export async function generateNdrfSitRep(analysisPayload) {
  const { data } = await apiClient.post("/api/generate-sitrep", analysisPayload);
  return data;
}

export async function simulateFloodTimeline(payload) {
  const { data } = await apiClient.post("/api/simulate-timeline", payload);
  return data;
}

export async function fetchModelBenchmarks() {
  const { data } = await apiClient.get("/api/benchmarks");
  return data;
}

export async function refreshOsmData(presetId) {
  const formData = new FormData();
  formData.append("preset_id", presetId);
  const { data } = await apiClient.post("/api/osm/refresh", formData);
  return data;
}

export async function fetchDataProvidersStatus() {
  const { data } = await apiClient.get("/api/data-providers/status");
  return data;
}

export async function searchDataProviders(payload) {
  const { data } = await apiClient.post("/api/data-providers/search", payload);
  return data;
}

export async function startDynamicSimulation(payload) {
  const { data } = await apiClient.post("/api/simulate-dynamic", payload);
  return data;
}

export async function fetchDynamicSimulationStatus(jobId) {
  const { data } = await apiClient.get(`/api/simulation-status/${jobId}`);
  return data;
}

export async function recalculateDynamicRoute(payload) {
  const { data } = await apiClient.post("/api/route-dynamic", payload);
  return data;
}

export async function exportSitRepMarkdown(payload) {
  const { data } = await apiClient.post("/api/sitrep/export-markdown", payload);
  return data.markdown;
}

export async function exportSitRepGeoJSON(payload) {
  const { data } = await apiClient.post("/api/sitrep/export-geojson", payload);
  return data;
}

export async function searchAOILocations(query) {
  const { data } = await apiClient.get("/api/aoi/search", { params: { q: query } });
  return data.locations || [];
}

export async function validateAOI(payload) {
  const { data } = await apiClient.post("/api/aoi/validate", payload);
  return data.aoi;
}

export async function analyzeDynamicAOI({ aoiName, bbox, geojsonGeometry, terrainType }) {
  const formData = new FormData();
  if (aoiName) formData.append("custom_name", aoiName);
  if (bbox) formData.append("bbox_json", JSON.stringify(bbox));
  const { data } = await apiClient.post("/api/analyze", formData);
  return data;
}

// Data Provider & Offline Caching API Client
export async function searchSatelliteCatalog(params) {
  const { data } = await apiClient.post("/api/data/satellite/search", params);
  return data;
}

export async function startDataDownloadJob(payload) {
  const { data } = await apiClient.post("/api/data/satellite/download", payload);
  return data;
}

export async function fetchDEMData(payload) {
  const { data } = await apiClient.post("/api/data/dem/fetch", payload);
  return data;
}

export async function fetchOSMData(payload) {
  const { data } = await apiClient.post("/api/data/osm/fetch", payload);
  return data;
}

export async function fetchRainfallData(payload) {
  const { data } = await apiClient.post("/api/data/rainfall/fetch", payload);
  return data;
}

export async function fetchGPMData(payload) {
  const { data } = await apiClient.post("/api/data/gpm/fetch", payload);
  return data;
}

export async function fetchDataJobStatus(jobId) {
  const { data } = await apiClient.get(`/api/data/jobs/${jobId}`);
  return data;
}

export async function fetchDataCache() {
  const { data } = await apiClient.get("/api/data/cache");
  return data;
}

export async function validateDataCache() {
  const { data } = await apiClient.post("/api/data/cache/validate");
  return data;
}

// Cyclone Scenario Simulation API Client
export async function fetchScenarioPresets() {
  const { data } = await apiClient.get("/api/scenarios/presets");
  return data;
}

export async function createScenario(payload) {
  const { data } = await apiClient.post("/api/scenarios/create", payload);
  return data;
}

export async function runScenario(scenarioId) {
  const { data } = await apiClient.post(`/api/scenarios/${scenarioId}/run`);
  return data;
}

export async function fetchScenarioStatus(scenarioId) {
  const { data } = await apiClient.get(`/api/scenarios/${scenarioId}/status`);
  return data;
}

export async function fetchScenarioResults(scenarioId) {
  const { data } = await apiClient.get(`/api/scenarios/${scenarioId}/results`);
  return data;
}

export async function fetchScenarioTimeline(scenarioId) {
  const { data } = await apiClient.get(`/api/scenarios/${scenarioId}/timeline`);
  return data;
}

export async function fetchScenarioReport(scenarioId) {
  const { data } = await apiClient.get(`/api/scenarios/${scenarioId}/report`);
  return data;
}

export async function recalculateScenarioRoute(scenarioId, payload) {
  const { data } = await apiClient.post(`/api/scenarios/${scenarioId}/routes/recalculate`, payload);
  return data;
}


