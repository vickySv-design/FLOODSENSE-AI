import React from "react";
import { Loader2 } from "lucide-react";

export default function LoadingState({
  message = "Processing Geospatial & AI Telemetry...",
  subtext = "Executing neural network inference and topological routing",
}) {
  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-slate-200 p-8 text-center flex flex-col items-center justify-center shadow-sm my-6">
      <div className="p-3 bg-blue-50 text-blue-600 rounded-full mb-3 border border-blue-100">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
      <h3 className="text-sm font-semibold text-slate-800 mb-1">{message}</h3>
      {subtext && (
        <p className="text-xs text-slate-500 max-w-sm mx-auto">{subtext}</p>
      )}
    </div>
  );
}

