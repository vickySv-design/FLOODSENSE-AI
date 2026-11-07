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

