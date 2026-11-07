import React, { useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  Cpu,
  Layers,
  MapPin,
  Navigation,
  Radar,
  Satellite,
  ShieldAlert,
  Waves,
  X,
  FileText,
  Activity,
} from "lucide-react";

export default function ProjectWalkthroughModal({ isOpen, onClose }) {
  const [currentStep, setCurrentStep] = useState(0);

  if (!isOpen) return null;

  const steps = [
    {
      title: "Step 1: Multi-Modal Satellite Ingestion",
      subtitle: "Optical & Radar Sensor Fusion (Sentinel-2 + Sentinel-1)",
      icon: Satellite,
      badge: "DATA ACQUISITION & CALIBRATION",
      badgeColor: "bg-cyan-500/15 text-cyan-300 border-cyan-500/40",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed font-sans">
          <p>
            During catastrophic monsoon floods, optical satellites (Sentinel-2, Landsat) suffer from{" "}
            <strong className="text-rose-400">&gt;80% cloud occlusion</strong>, hiding ground inundation.
            Conversely, synthetic aperture radar (Sentinel-1 C-Band SAR) penetrates cloud decks and rain 24/7,
            measuring surface specular scattering.
          </p>
          <div className="grid grid-cols-2 gap-2 p-2.5 rounded bg-slate-950 border border-slate-800 font-mono text-[11px]">
            <div>
              <span className="text-cyan-400 font-semibold">Stream 1 (Pre-Flood):</span>
              <ul className="list-disc list-inside text-slate-400 mt-1 space-y-0.5">
                <li>Sentinel-2 MSI (10m res)</li>
                <li>4 Channels: Red, Green, Blue, NIR</li>
                <li>Pre-event baseline &amp; NDWI reference</li>
              </ul>
            </div>
            <div>
              <span className="text-rose-400 font-semibold">Stream 2 (Post-Flood):</span>
              <ul className="list-disc list-inside text-slate-400 mt-1 space-y-0.5">
                <li>Sentinel-1 IW GRD (10m res)</li>
                <li>2 Channels: Dual-Pol VV + VH (dB)</li>
                <li>All-weather active flood backscatter drop</li>
              </ul>
            </div>
          </div>
        </div>
      ),
    },
    {
      title: "Step 2: Dual-Stream Siamese Attention U-Net",
      subtitle: "Cross-Modal Attention Fusion (CMAF) & Decoder Attention Gates",
      icon: Cpu,
      badge: "DEEP LEARNING ARCHITECTURE",
      badgeColor: "bg-purple-500/15 text-purple-300 border-purple-500/40",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed font-sans">
          <p>
            Early concatenation forces convolutional kernels to jointly convolve surface reflectance with
            microwave radar backscatter. Our architecture isolates optical and radar signals in dedicated encoder
            branches, performing Cross-Modal Attention Fusion (CMAF) at 4 spatial scales (64, 128, 256, 512 channels).
          </p>
          <div className="p-2.5 rounded bg-slate-950 border border-slate-800 font-mono text-[11px] space-y-1">
            <div className="text-purple-300 font-semibold">Key Architectural Innovations:</div>
            <div className="text-slate-400">1. Squeeze-and-Excitation (SE) channel calibration per modality</div>
            <div className="text-slate-400">2. Spatial cross-attention gating dynamically suppressing cloud noise</div>
            <div className="text-slate-400">3. Symmetrical transposed-convolution decoder with skip-attention gates</div>
            <div className="text-slate-400">4. Parameter footprint: ~17.4M weights (optimal for real-time edge inference)</div>
          </div>
        </div>
      ),
    },
    {
      title: "Step 3: Physics-Guided Remote Sensing Priors",
      subtitle: "Normalized Difference Water Index (NDWI) & SAR Specular Prior",
      icon: Radar,
      badge: "HYBRID AI / PHYSICAL PRIORS",
      badgeColor: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed font-sans">
          <p>
            Pure deep learning models can produce spurious edge artifacts. FloodSense-AI combines model-derived
            probabilities with strict physical principles from remote sensing science:
          </p>
          <div className="p-2.5 rounded bg-slate-950 border border-slate-800 font-mono text-[11px] space-y-1.5">
            <div>
              <span className="text-cyan-300">NDWI Formula:</span>{" "}
              <code className="text-emerald-400">(Green - NIR) / (Green + NIR) &gt; 0.10</code>
            </div>
            <div>
              <span className="text-rose-300">SAR Specular Drop:</span>{" "}
              <code className="text-amber-400">Delta_VV = VV_post - VV_pre &lt; -3.5 dB</code>
            </div>
            <p className="text-slate-400 text-[10px] pt-1">
              Smooth open water acts as a specular mirror for microwaves, causing severe backscatter drop.
              Merging deep features with NDWI removes radar shadow false alarms.
            </p>
          </div>
        </div>
      ),
    },
    {
      title: "Step 4: Real OpenStreetMap Vector Geoprocessing",
      subtitle: "5,000+ Real Road Segments & 27,000+ Building Polygons",
      icon: MapPin,
      badge: "VECTOR GIS TOPOLOGY",
      badgeColor: "bg-blue-500/15 text-blue-300 border-blue-500/40",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed font-sans">
          <p>
            Rather than synthetic grids, FloodSense-AI connects to the OpenStreetMap Overpass and OSM 0.6 APIs
            to ingest actual road polylines (motorways, primary, secondary, residential) and building footprint polygons.
          </p>
          <div className="grid grid-cols-2 gap-2 p-2.5 rounded bg-slate-950 border border-slate-800 font-mono text-[11px]">
            <div>
              <span className="text-amber-400 font-semibold">Building Damage:</span>
              <ul className="list-disc list-inside text-slate-400 mt-1 space-y-0.5">
                <li>Safe Dry: h &lt; 0.05m</li>
                <li>Waterlogged: 0.05m - 0.30m</li>
                <li>Moderate: 0.30m - 1.00m</li>
                <li>Severe Submerged: h &ge; 1.00m</li>
              </ul>
            </div>
            <div>
              <span className="text-rose-400 font-semibold">Road Hazards:</span>
              <ul className="list-disc list-inside text-slate-400 mt-1 space-y-0.5">
                <li>OPEN: Passable (all vehicles)</li>
                <li>CAUTION: 4x4 / High Clearance</li>
                <li>HIGH RISK: Specialist unimogs</li>
                <li>BLOCKED: Requires boats</li>
              </ul>
            </div>
          </div>
        </div>
      ),
    },
    {
      title: "Step 5: 2D Dynamic Time-Stepped Flood Simulation",
      subtitle: "Mass-Conservative Shallow-Water Diffusion (0.00% Mass Error)",
      icon: Waves,
      badge: "HYDROLOGICAL MODELLING",
      badgeColor: "bg-indigo-500/15 text-indigo-300 border-indigo-500/40",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed font-sans">
          <p>
            Replaces visual morphological dilation with a genuine 2D grid-based shallow-water diffusive solver
            governed by the continuity and momentum equations:
          </p>
          <div className="p-2.5 rounded bg-slate-950 border border-slate-800 font-mono text-[11px] space-y-1">
            <div className="text-cyan-300 font-semibold">Governing Equation:</div>
            <code className="text-emerald-400">dh/dt + dqx/dx + dqy/dy = R(t) - I(t) - D(t)</code>
            <div className="text-slate-400 text-[10px] pt-1">
              - <strong className="text-slate-200">DEM Slopes:</strong> Hydro-conditioned DEM with carved river channels.<br />
              - <strong className="text-slate-200">Manning Friction:</strong> Roughness n=0.035 for urban floodplains.<br />
              - <strong className="text-slate-200">Strict Conservation:</strong> Adaptive flux limiter guarantees exact mass balance error &lt; 0.01%.
            </div>
          </div>
        </div>
      ),
    },
    {
      title: "Step 6: Dynamic NetworkX A* Rescue Routing",
      subtitle: "Sub-Second Shortest Safe Evacuation Corridors",
      icon: Navigation,
      badge: "EMERGENCY LOGISTICS",
      badgeColor: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed font-sans">
          <p>
            Connects emergency bases to isolated shelters through the real OSM graph using the NetworkX A* algorithm.
            Edges are dynamically weighted by flood depths and vehicle capabilities.
          </p>
          <div className="p-2.5 rounded bg-slate-950 border border-slate-800 font-mono text-[11px] space-y-1">
            <div className="text-emerald-400 font-semibold">Heuristic &amp; Safety Cost:</div>
            <div className="text-slate-300">Cost(u, v) = Distance_km &times; Water_Penalty_Multiplier</div>
            <div className="text-slate-400 text-[10px]">
              If water exceeds vehicle clearance (e.g., 0.8m for standard trucks), edge weight becomes infinity,
              forcing the router to discover dry peripheral bypass routes or signal NO_SAFE_ROUTE_AVAILABLE.
            </div>
          </div>
        </div>
      ),
    },
    {
      title: "Step 7: Automated NDRF AI Situation Report",
      subtitle: "Tactical Flash Bulletins & Multi-Format Spatial Export",
      icon: FileText,
      badge: "EXECUTIVE REPORTING",
      badgeColor: "bg-rose-500/15 text-rose-300 border-rose-500/40",
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed font-sans">
          <p>
            Synthesizes raw spatial telemetry into an official emergency briefing compliant with National Disaster
            Response Force (NDRF) and State Disaster Management Authority (SDMA) operational protocols.
          </p>
          <div className="p-2.5 rounded bg-slate-950 border border-slate-800 font-mono text-[11px] space-y-1">
            <div className="text-rose-400 font-semibold">Automated Deliverables:</div>
            <div className="text-slate-300">1. Executive Hydrological Summary &amp; Classification Level</div>
            <div className="text-slate-300">2. Resource Allocations (Inflatable boats, medical teams, buses)</div>
            <div className="text-slate-300">3. 4 Concrete Directives (Alpha, Bravo, Charlie, Delta)</div>
            <div className="text-slate-300">4. Export formats: Markdown (.md), GeoJSON (.geojson), and Printable PDF</div>
          </div>
        </div>
      ),
    },
  ];

  const current = steps[currentStep];
  const StepIcon = current.icon;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-md flex items-center justify-center p-4">
      <div className="w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-5 py-4 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-cyan-500/15 text-cyan-400 border border-cyan-500/30">
              <StepIcon className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-bold text-slate-100">PROJECT REVIEW &amp; DEFENSE GUIDE</span>
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${current.badgeColor}`}>
                  {current.badge}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                Stage {currentStep + 1} of {steps.length}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-100 border border-slate-800 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 space-y-4">
          <div>
            <h3 className="text-base font-semibold text-slate-100">{current.title}</h3>
            <p className="text-xs text-cyan-400 font-mono mt-0.5">{current.subtitle}</p>
          </div>

          <div className="min-h-[170px]">{current.content}</div>

          {/* Stepper Dots */}
          <div className="flex items-center justify-center gap-1.5 pt-2">
            {steps.map((_, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setCurrentStep(idx)}
                className={`w-2.5 h-2.5 rounded-full transition-all ${
                  idx === currentStep ? "w-6 bg-cyan-400" : "bg-slate-700 hover:bg-slate-600"
                }`}
                title={`Go to step ${idx + 1}`}
              />
            ))}
          </div>
        </div>

        {/* Footer Navigation */}
        <div className="px-5 py-3.5 bg-slate-950 border-t border-slate-800 flex items-center justify-between">
          <button
            type="button"
            onClick={() => setCurrentStep((prev) => Math.max(0, prev - 1))}
            disabled={currentStep === 0}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-mono font-semibold bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 transition disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-4 h-4" /> Previous
          </button>

          {currentStep < steps.length - 1 ? (
            <button
              type="button"
              onClick={() => setCurrentStep((prev) => Math.min(steps.length - 1, prev + 1))}
              className="inline-flex items-center gap-1 px-4 py-1.5 rounded-md text-xs font-mono font-bold bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow transition"
            >
              Next <ChevronRight className="w-4 h-4" />
            </button>
          ) : (
            <button
              type="button"
              onClick={onClose}
              className="inline-flex items-center gap-1 px-4 py-1.5 rounded-md text-xs font-mono font-bold bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow transition"
            >
              <CheckCircle2 className="w-4 h-4" /> Finish Walkthrough
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
