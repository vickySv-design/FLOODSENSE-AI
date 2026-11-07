import React from "react";

export default function StatusIndicator({
  status = "neutral", // "success" | "warning" | "danger" | "info" | "neutral"
  label,
  pulse = false,
  size = "md", // "sm" | "md"
}) {
  const styles = {
    success: {
      dot: "bg-emerald-500",
      badge: "bg-emerald-50 text-emerald-700 border-emerald-200",
    },
    warning: {
      dot: "bg-amber-500",
      badge: "bg-amber-50 text-amber-700 border-amber-200",
    },
    danger: {
      dot: "bg-red-500",
      badge: "bg-red-50 text-red-700 border-red-200",
    },
    info: {
      dot: "bg-blue-500",
      badge: "bg-blue-50 text-blue-700 border-blue-200",
    },
    cyan: {
      dot: "bg-cyan-500",
      badge: "bg-cyan-50 text-cyan-700 border-cyan-200",
    },
    neutral: {
      dot: "bg-slate-400",
      badge: "bg-slate-100 text-slate-700 border-slate-200",
    },
  };

  const current = styles[status] || styles.neutral;
  const sizeClasses = size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs";

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full font-medium border ${sizeClasses} ${current.badge}`}>
      <span className="relative flex h-2 w-2">
        {pulse && (
          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${current.dot} opacity-75`} />
        )}
        <span className={`relative inline-flex rounded-full h-2 w-2 ${current.dot}`} />
      </span>
      {label}
    </span>
  );
}

