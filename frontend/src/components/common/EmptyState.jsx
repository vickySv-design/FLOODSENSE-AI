import React from "react";
import { FolderSearch } from "lucide-react";

export default function EmptyState({
  icon: Icon = FolderSearch,
  title = "No Data Available",
  description = "No disaster analysis data or simulation results have been generated yet.",
  action,
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-8 text-center flex flex-col items-center justify-center max-w-lg mx-auto shadow-sm my-6">
      <div className="p-3 bg-slate-50 text-slate-500 rounded-full mb-3 border border-slate-100">
        <Icon className="w-8 h-8 text-slate-400" />
      </div>
      <h3 className="text-base font-semibold text-slate-800 mb-1">{title}</h3>
      <p className="text-sm text-slate-500 leading-relaxed mb-4">{description}</p>
      {action && <div>{action}</div>}
    </div>
  );
}
