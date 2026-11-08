import React, { useEffect, useState } from "react";
import AppLayout from "./components/layout/AppLayout.jsx";
import DashboardPage from "./pages/DashboardPage.jsx";
import FloodSensePipelinePage from "./pages/FloodSensePipelinePage.jsx";
import SatelliteAnalysisPage from "./pages/SatelliteAnalysisPage.jsx";
import FloodSimulationPage from "./pages/FloodSimulationPage.jsx";
import RescueInfrastructurePage from "./pages/RescueInfrastructurePage.jsx";
import DisasterReportsPage from "./pages/DisasterReportsPage.jsx";
import DataSourcesPage from "./pages/DataSourcesPage.jsx";
import SettingsPage from "./pages/SettingsPage.jsx";

// Global Modals preserved
import SitRepModal from "./components/SitRepModal.jsx";
import ProjectWalkthroughModal from "./components/ProjectWalkthroughModal.jsx";
import UniversalAOIModal from "./components/UniversalAOIModal.jsx";

import {
  analyzeCustomUpload,
  analyzeDynamicAOI,
  analyzePresetScenario,
  fetchAndAnalyzeGEETiles,
  fetchDisasterPresets,
  fetchHealthStatus,
  generateNdrfSitRep,
} from "./services/api.js";

const VALID_PAGES = [
  "dashboard",
  "pipeline",
  "satellite-analysis",
  "flood-simulation",
  "rescue",
  "reports",
  "data-sources",
  "settings",
];

export default function App() {
  // Hash-based navigation with back/forward history support
  const getInitialPage = () => {
    const hash = window.location.hash.replace("#", "");
    return VALID_PAGES.includes(hash) ? hash : "dashboard";
  };

  const [activePage, setActivePage] = useState(getInitialPage);
  const [health, setHealth] = useState(null);
  const [presets, setPresets] = useState([]);
  const [selectedPresetId, setSelectedPresetId] = useState("chennai_monsoon_2026");
  const [analysisData, setAnalysisData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState(null);

  // Global Modals
  const [isSitRepOpen, setIsSitRepOpen] = useState(false);
  const [sitrepData, setSitrepData] = useState(null);
  const [isGeneratingSitRep, setIsGeneratingSitRep] = useState(false);
  const [isWalkthroughOpen, setIsWalkthroughOpen] = useState(false);
  const [isAOIModalOpen, setIsAOIModalOpen] = useState(false);

  // Synchronize hash on page change and popstate
  const handleNavigate = (pageId) => {
    if (VALID_PAGES.includes(pageId)) {
      setActivePage(pageId);
      window.location.hash = pageId;
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace("#", "");
      if (VALID_PAGES.includes(hash) && hash !== activePage) {
        setActivePage(hash);
      }
    };
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, [activePage]);

  // Initial Load & Telemetry Boot
  useEffect(() => {
    let mounted = true;
    async function bootMissionControl() {
      setIsLoading(true);
      setErrorMsg(null);
      try {
        const [healthRes, presetsRes] = await Promise.all([
          fetchHealthStatus(),
          fetchDisasterPresets(),
        ]);
        if (!mounted) return;
        setHealth(healthRes);
        setPresets(presetsRes.presets || []);
        const defaultId = presetsRes.default_preset_id || "chennai_monsoon_2026";
        setSelectedPresetId(defaultId);

        const initialAnalysis = await analyzePresetScenario(defaultId);
        if (mounted) {
          setAnalysisData(initialAnalysis);
        }
      } catch (err) {
        if (mounted) {
          setErrorMsg(
            err?.response?.data?.detail ||
              err?.message ||
              "Failed to connect to FloodSense-AI Backend at http://localhost:8000"
          );
        }
      } finally {
        if (mounted) {
          setIsLoading(false);
        }
      }
    }
    bootMissionControl();
    return () => {
      mounted = false;
    };
  }, []);

  const handleSelectPreset = async (presetId) => {
    setSelectedPresetId(presetId);
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const result = await analyzePresetScenario(presetId);
      setAnalysisData(result);
    } catch (err) {
      setErrorMsg(
        err?.response?.data?.detail ||
          err?.message ||
          "Error analyzing disaster preset."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadCustomPair = async (uploadPayload) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const result = await analyzeCustomUpload(uploadPayload);
      setSelectedPresetId("custom_upload_mission");
      setAnalysisData(result);
    } catch (err) {
      setErrorMsg(
        err?.response?.data?.detail ||
          err?.message ||
          "Error processing custom uploaded satellite tiles."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleFetchGEE = async (geePayload) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const result = await fetchAndAnalyzeGEETiles(geePayload);
      setSelectedPresetId("gee_live_acquisition");
      setAnalysisData(result);
    } catch (err) {
      setErrorMsg(
        err?.response?.data?.detail ||
          err?.message ||
          "Error fetching Copernicus Sentinel-1/2 tiles."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleApplyAOI = async (aoiConfig) => {
    if (aoiConfig.preset_id) {
      await handleSelectPreset(aoiConfig.preset_id);
      return;
    }
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const result = await analyzeDynamicAOI({
        aoiName: aoiConfig.name,
        bbox: aoiConfig.bbox,
        terrainType: aoiConfig.terrain_type,
      });
      setSelectedPresetId(result.scenario?.id || "dynamic_aoi");
      setAnalysisData(result);
    } catch (err) {
      setErrorMsg(
        err?.response?.data?.detail ||
          err?.message ||
          "Error analyzing custom Area of Interest."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleOpenSitRep = async () => {
    if (!analysisData) return;
    setIsSitRepOpen(true);
    setIsGeneratingSitRep(true);
    try {
      const res = await generateNdrfSitRep(analysisData);
      setSitrepData(res);
    } catch (err) {
      setErrorMsg(
        err?.response?.data?.detail ||
          err?.message ||
          "Failed to generate NDRF Situation Report."
      );
    } finally {
      setIsGeneratingSitRep(false);
    }
  };

  return (
    <AppLayout
      activePage={activePage}
      onNavigate={handleNavigate}
      analysisData={analysisData}
      health={health}
      isLoading={isLoading}
      onOpenAOIModal={() => setIsAOIModalOpen(true)}
      onOpenSitRep={handleOpenSitRep}
      onOpenWalkthrough={() => setIsWalkthroughOpen(true)}
    >
      {/* Global Error Notice if backend connection fails */}
      {errorMsg && (
        <div className="bg-red-50 border-b border-red-200 px-6 py-3 text-xs text-red-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-red-600 inline-block" />
            <span>{errorMsg}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorMsg(null)}
            className="text-red-700 font-semibold hover:underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Page Routing */}
      {activePage === "dashboard" && (
        <DashboardPage
          analysisData={analysisData}
          isLoading={isLoading}
          onNavigate={handleNavigate}
          onOpenAOIModal={() => setIsAOIModalOpen(true)}
          onOpenSitRep={handleOpenSitRep}
          onSelectPreset={handleSelectPreset}
          presets={presets}
          selectedPresetId={selectedPresetId}
        />
      )}

      {activePage === "pipeline" && (
        <FloodSensePipelinePage
          analysisData={analysisData}
          presets={presets}
        />
      )}

      {activePage === "satellite-analysis" && (
        <SatelliteAnalysisPage
          analysisData={analysisData}
          presets={presets}
          selectedPresetId={selectedPresetId}
          onSelectPreset={handleSelectPreset}
          onUploadCustomPair={handleUploadCustomPair}
          onFetchGEE={handleFetchGEE}
          onOpenAOIModal={() => setIsAOIModalOpen(true)}
          isLoading={isLoading}
          errorMsg={errorMsg}
        />
      )}

      {activePage === "flood-simulation" && (
        <FloodSimulationPage
          analysisData={analysisData}
          currentAOI={analysisData?.scenario}
          onNavigate={handleNavigate}
        />
      )}

      {activePage === "rescue" && (
        <RescueInfrastructurePage
          analysisData={analysisData}
          isLoading={isLoading}
          onOpenSitRep={handleOpenSitRep}
        />
      )}

      {activePage === "reports" && (
        <DisasterReportsPage
          analysisData={analysisData}
          isLoading={isLoading}
          sitrepData={sitrepData}
        />
      )}

      {activePage === "data-sources" && (
        <DataSourcesPage currentAOI={analysisData?.scenario} />
      )}

      {activePage === "settings" && (
        <SettingsPage
          analysisData={analysisData}
          health={health}
          presets={presets}
          selectedPresetId={selectedPresetId}
          onSelectPreset={handleSelectPreset}
          onOpenWalkthrough={() => setIsWalkthroughOpen(true)}
        />
      )}

      {/* Global Preserved Modals */}
      <SitRepModal
        isOpen={isSitRepOpen}
        onClose={() => setIsSitRepOpen(false)}
        sitrepData={sitrepData}
        isGenerating={isGeneratingSitRep}
      />

      <ProjectWalkthroughModal
        isOpen={isWalkthroughOpen}
        onClose={() => setIsWalkthroughOpen(false)}
      />

      <UniversalAOIModal
        isOpen={isAOIModalOpen}
        onClose={() => setIsAOIModalOpen(false)}
        currentAOI={analysisData?.scenario}
        presets={presets}
        onApplyAOI={handleApplyAOI}
      />
    </AppLayout>
  );
}
