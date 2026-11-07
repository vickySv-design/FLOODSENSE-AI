import React, { useRef, useState } from "react";
import { Eye, MoveHorizontal, Satellite, Sparkles } from "lucide-react";

export default function SplitCompareSlider({ analysisData }) {
  const [sliderPos, setSliderPos] = useState(50);
  const [rightLayerKey, setRightLayerKey] = useState("flood_overlay");
  const containerRef = useRef(null);

  if (!analysisData?.visual_layers) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-950 text-slate-400 font-mono text-sm">
        Rendering Multi-Spectral Raster Layers...
      </div>
    );
  }

  const { visual_layers, scenario, engine } = analysisData;
  const bbox = scenario?.bbox || [13.01, 80.22, 13.06, 80.27];

  const rightLayerOptions = [
    {
      key: "flood_overlay",
      title: "SAR + Semantic Flood Mask",
      subtitle: "Class 1 (Cyan) + Class 2 Flood (Red) + 100m Buffer",
      badge: "AI Fusion Output",
    },
    {
      key: "sar_composite",
      title: "Sentinel-1 SAR (Lee-Sigma Filtered)",
      subtitle: "Dual-Pol VV/VH Specular Backscatter Composite",
      badge: "Post-Event Radar",
    },
    {
      key: "ndwi_heatmap",
      title: "NDWI Hydrological Index",
      subtitle: "(Green - NIR) / (Green + NIR) Water Surface Map",
      badge: "Pre-Event Baseline",
    },
    {
      key: "attention_heatmap",
      title: "XAI Attention Gate Saliency",
      subtitle: "Multi-Scale Cross-Modal Attention Activation Map",
      badge: "Explainable AI",
    },
  ];

  const activeRightMeta =
    rightLayerOptions.find((o) => o.key === rightLayerKey) ||
    rightLayerOptions[0];

  const updateSliderFromClientX = (clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const relX = clientX - rect.left;
    const pct = Math.max(2, Math.min(98, (relX / rect.width) * 100));
    setSliderPos(pct);
  };

  const handleMouseMove = (e) => {
    if (e.buttons !== 1) return;
    updateSliderFromClientX(e.clientX);
  };

  const handleTouchMove = (e) => {
    if (e.touches?.[0]) {
      updateSliderFromClientX(e.touches[0].clientX);
    }
  };

  return (
    <div className="w-full h-full bg-slate-950 flex flex-col lg:flex-row gap-3 p-3 overflow-y-auto">
      {/* Main Interactive Split Curtain Viewport */}
      <div className="flex-1 flex flex-col bg-slate-900/70 border border-slate-800 rounded-xl overflow-hidden">
        <div className="px-4 py-2.5 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2 bg-slate-900">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-mono font-bold text-slate-100">
              MULTI-MODAL CURTAIN COMPARATOR // {scenario?.name}
            </span>
          </div>
          <div className="flex items-center gap-3 text-[11px] font-mono">
            <span className="text-cyan-300">
              LEFT: Pre-Flood Sentinel-2 Optical RGB ({scenario?.pre_date})
            </span>
            <span className="text-slate-600">vs.</span>
            <span className="text-rose-400">
              RIGHT: {activeRightMeta.title} ({scenario?.post_date})
            </span>
          </div>
        </div>

        {/* Interactive Image Comparison Stage */}
        <div className="flex-1 relative flex items-center justify-center p-3 bg-slate-950 select-none overflow-hidden">
          <div
            ref={containerRef}
            onMouseDown={(e) => updateSliderFromClientX(e.clientX)}
            onMouseMove={handleMouseMove}
            onTouchMove={handleTouchMove}
            className="relative w-full max-w-[540px] aspect-square rounded-lg overflow-hidden border border-slate-700 shadow-2xl cursor-ew-resize"
          >
            {/* Base (Right Side) Image: Post-Disaster SAR / Flood Overlay / XAI Heatmap */}
            <img
              src={visual_layers[rightLayerKey]}
              alt={activeRightMeta.title}
              draggable={false}
              className="absolute inset-0 w-full h-full object-cover"
            />

            {/* Clipped (Left Side) Image: Pre-Disaster Sentinel-2 True-Color Optical RGB */}
            <div
              className="absolute inset-0 overflow-hidden"
              style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
            >
              <img
                src={visual_layers.optical_rgb}
                alt="Pre-Disaster Sentinel-2 Optical RGB"
                draggable={false}
                className="w-full h-full object-cover"
              />
            </div>

            {/* Vertical Curtain Divider Line & Handle */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 shadow-[0_0_12px_rgba(34,211,238,0.95)] pointer-events-none"
              style={{ left: `${sliderPos}%` }}
            >
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-slate-900/95 border-2 border-cyan-400 text-cyan-300 flex items-center justify-center shadow-lg">
                <MoveHorizontal className="w-4 h-4" />
              </div>
            </div>

            {/* Top-Left Copernicus Sentinel-2 HUD Telemetry Stamp */}
            <div className="absolute top-2.5 left-2.5 px-2.5 py-1.5 rounded bg-slate-950/90 border border-cyan-500/40 text-[10px] font-mono text-cyan-300 pointer-events-none space-y-0.5 shadow-lg">
              <div className="font-bold flex items-center gap-1">
                <Satellite className="w-3 h-3 text-cyan-400" />
                SENSOR: S2B MSI (B4/B3/B2/B8)
              </div>
              <div className="text-slate-300">
                GSD: 10.0m | PASS: {scenario?.pre_date} 05:18 UTC
              </div>
              <div className="text-emerald-300">
                QA60 CLOUD COVER: 16.0% | CLAHE: ACTIVE
              </div>
            </div>

            {/* Top-Right Copernicus Sentinel-1 SAR HUD Telemetry Stamp */}
            <div className="absolute top-2.5 right-2.5 px-2.5 py-1.5 rounded bg-slate-950/90 border border-rose-500/40 text-[10px] font-mono text-rose-300 pointer-events-none space-y-0.5 text-right shadow-lg">
              <div className="font-bold">
                SENSOR: S1A C-Band SAR (VV+VH)
              </div>
              <div className="text-slate-300">
                ORBIT: DESCENDING | PASS: {scenario?.post_date} 00:42 UTC
              </div>
              <div className="text-amber-300">
                MODE: {activeRightMeta.badge} | GSD: 10.0m
              </div>
            </div>

            {/* Bottom-Left SAR Speckle & Radiometric Telemetry Stamp */}
            <div className="absolute bottom-2.5 left-2.5 px-2 py-1 rounded bg-slate-950/85 border border-slate-700/80 text-[9px] font-mono text-slate-300 pointer-events-none">
              LEE-SIGMA FILTER: 5x5 (ENL=4.2) | WATER CUTOFF: -16.0 dB
            </div>

            {/* Bottom-Right Spatial Coordinates & Inference Latency Stamp */}
            <div className="absolute bottom-2.5 right-2.5 px-2 py-1 rounded bg-slate-950/85 border border-slate-700/80 text-[9px] font-mono text-cyan-300 pointer-events-none">
              AOI: [{bbox[0].toFixed(2)}°N, {bbox[1].toFixed(2)}°E] |{" "}
              {engine?.latency_ms ?? 38}ms
            </div>
          </div>
        </div>

        {/* Bottom Slider Bar Control */}
        <div className="px-4 py-2.5 bg-slate-900 border-t border-slate-800 flex items-center gap-4">
          <span className="text-[11px] font-mono text-cyan-300 shrink-0">
            0% (Post-Event Only)
          </span>
          <input
            type="range"
            min="2"
            max="98"
            value={sliderPos}
            onChange={(e) => setSliderPos(Number(e.target.value))}
            className="w-full accent-cyan-400 cursor-pointer"
          />
          <span className="text-[11px] font-mono text-emerald-300 shrink-0">
            100% (Pre-Event Only)
          </span>
        </div>
      </div>

      {/* Right Column: Quick-Select Multi-Spectral Layer Inspector Cards */}
      <div className="w-full lg:w-72 shrink-0 flex flex-col gap-2.5">
        <div className="px-3 py-2 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between">
          <span className="text-xs font-mono font-bold text-slate-200 flex items-center gap-1.5">
            <Eye className="w-3.5 h-3.5 text-cyan-400" />
            Right Curtain Layer
          </span>
          <span className="text-[10px] font-mono text-slate-400">4 Rasters</span>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-1 gap-2.5 flex-1">
          {rightLayerOptions.map((opt) => {
            const isSelected = rightLayerKey === opt.key;
            return (
              <button
                key={opt.key}
                type="button"
                onClick={() => setRightLayerKey(opt.key)}
                className={`text-left p-2 rounded-lg border transition flex items-center gap-2.5 ${
                  isSelected
                    ? "bg-cyan-950/40 border-cyan-400 shadow-md"
                    : "bg-slate-900/80 border-slate-800 hover:border-slate-700"
                }`}
              >
                <img
                  src={visual_layers[opt.key]}
                  alt={opt.title}
                  className="w-16 h-16 rounded object-cover border border-slate-700 shrink-0"
                />
                <div className="min-w-0">
                  <span className="inline-block text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-slate-950 text-cyan-300 border border-slate-800 mb-1">
                    {opt.badge}
                  </span>
                  <div className="text-xs font-bold text-slate-100 truncate">
                    {opt.title}
                  </div>
                  <div className="text-[10px] text-slate-400 line-clamp-2 mt-0.5">
                    {opt.subtitle}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
