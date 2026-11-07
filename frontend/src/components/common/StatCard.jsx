import React from "react";

export default function StatCard({
  title,
  value,
  unit,
  subvalue,
  icon: Icon,
  trend,
  color = "blue", // "blue" | "red" | "amber" | "emerald" | "cyan"
  footer,
}) {
  const colorMap = {
    blue: {
      bg: "bg-blue-50",
      text: "text-blue-600",
      border: "border-blue-100",
      accent: "border-t-blue-600",
    },
    red: {
      bg: "bg-red-50",
      text: "text-red-600",
      border: "border-red-100",
      accent: "border-t-red-600",
    },
    amber: {
      bg: "bg-amber-50",
      text: "text-amber-600",
      border: "border-amber-100",
      accent: "border-t-amber-600",
    },
    emerald: {
      bg: "bg-emerald-50",
      text: "text-emerald-600",
      border: "border-emerald-100",
      accent: "border-t-emerald-600",
    },
    cyan: {
      bg: "bg-cyan-50",
      text: "text-cyan-600",
      border: "border-cyan-100",
      accent: "border-t-cyan-600",
    },
  };

  const scheme = colorMap[color] || colorMap.blue;

  return (
    <div className={`bg-white rounded-xl border border-slate-200 border-t-4 ${scheme.accent} p-4 shadow-sm hover:shadow-md transition flex flex-col justify-between`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            {title}
          </span>
          <div className="mt-1 flex items-baseline gap-1.5 flex-wrap">
            <span className="text-2xl font-bold text-slate-900 tracking-tight">
              {value ?? "--"}
            </span>
            {unit && (
              <span className="text-sm font-medium text-slate-500">
                {unit}
              </span>
            )}
          </div>
          {subvalue && (
            <p className="text-xs text-slate-500 mt-0.5">
              {subvalue}
            </p>
          )}
        </div>

        {Icon && (
          <div className={`p-2.5 rounded-lg ${scheme.bg} ${scheme.text} ${scheme.border} border shrink-0`}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>

      {footer && (
        <div className="mt-3 pt-3 border-t border-slate-100 text-xs text-slate-600 flex items-center justify-between">
          {footer}
        </div>
      )}
    </div>
  );
}
