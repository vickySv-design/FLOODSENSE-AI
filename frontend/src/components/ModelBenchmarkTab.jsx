import React, { useEffect, useState } from "react";
import { Award, Cpu, Layers, ShieldCheck } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchModelBenchmarks } from "../services/api.js";

export default function ModelBenchmarkTab() {
  const [benchData, setBenchData] = useState(null);

  useEffect(() => {
    let mounted = true;
    fetchModelBenchmarks()
      .then((res) => {
        if (mounted) setBenchData(res);
      })
      .catch((err) => console.error("Failed to load benchmarks:", err));
    return () => {
      mounted = false;
    };
  }, []);

  if (!benchData) {
    return (
      <div className="p-4 text-center text-xs font-mono text-slate-400">
        Loading Sen1Floods11 Architecture Ablation Benchmarks...
      </div>
    );
  }

  const { architectures, ablation_study, viva_defense_summary } = benchData;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
      {/* Left: Multi-Bar Recharts Comparison (mIoU, F1, Precision, Recall) */}
      <div className="lg:col-span-6 p-3 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-mono font-bold text-cyan-300 flex items-center gap-1.5">
            <Award className="w-4 h-4 text-amber-400" />
            Sen1Floods11 Benchmark (&gt;80% Monsoon Cloud Cover)
          </span>
          <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
            Ours: 89.2% mIoU | 94.1% F1
          </span>
        </div>

        <div className="h-48 w-full flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={architectures}
              margin={{ top: 6, right: 10, left: -16, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="short_name"
                tick={{ fill: "#cbd5e1", fontSize: 10 }}
                axisLine={{ stroke: "#334155" }}
              />
              <YAxis
                domain={[40, 100]}
                tickFormatter={(v) => `${v}%`}
                tick={{ fill: "#94a3b8", fontSize: 10 }}
                axisLine={{ stroke: "#334155" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  fontSize: "11px",
                  borderRadius: "6px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              <Bar
                dataKey="miou_pct"
                name="mIoU (%)"
                fill="#06b6d4"
                radius={[3, 3, 0, 0]}
              />
              <Bar
                dataKey="f1_pct"
                name="Dice / F1 (%)"
                fill="#10b981"
                radius={[3, 3, 0, 0]}
              />
              <Bar
                dataKey="precision_pct"
                name="Precision (%)"
                fill="#f59e0b"
                radius={[3, 3, 0, 0]}
              />
              <Bar
                dataKey="recall_pct"
                name="Recall (%)"
                fill="#a855f7"
                radius={[3, 3, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Right: Architecture Comparison Table & Viva Defense Summary */}
      <div className="lg:col-span-6 p-3 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col justify-between space-y-2.5">
        <div className="overflow-x-auto">
          <div className="text-xs font-mono font-bold text-slate-200 mb-1.5 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              Architecture & Ablation Telemetry (Viva Defense)
            </span>
            <span className="text-[10px] text-slate-400">
              Params: 31.4M | ~38ms
            </span>
          </div>

          <table className="w-full text-left border-collapse font-mono text-[11px]">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-1 pr-2">Architecture</th>
                <th className="py-1 px-2">mIoU</th>
                <th className="py-1 px-2">F1</th>
                <th className="py-1 px-2">Params</th>
                <th className="py-1 pl-2">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {architectures.map((arch) => {
                const isOurs = arch.id.includes("proposed");
                return (
                  <tr
                    key={arch.id}
                    className={
                      isOurs
                        ? "bg-cyan-500/10 text-cyan-200 font-bold"
                        : "text-slate-300"
                    }
                  >
                    <td className="py-1.5 pr-2 truncate max-w-[190px]">
                      {arch.model_name}
                    </td>
                    <td className="py-1.5 px-2 text-cyan-300">
                      {(arch.miou * 100).toFixed(1)}%
                    </td>
                    <td className="py-1.5 px-2 text-emerald-300">
                      {(arch.f1_score * 100).toFixed(1)}%
                    </td>
                    <td className="py-1.5 px-2">{arch.params_m}M</td>
                    <td className="py-1.5 pl-2">{arch.inference_ms}ms</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Component-Wise Ablation Strip */}
        <div className="flex flex-wrap items-center gap-1.5 pt-1">
          {ablation_study?.map((step, i) => (
            <span
              key={i}
              className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300"
            >
              {step.configuration.split("+").pop()?.trim()}:{" "}
              <strong className="text-emerald-400">{step.gain_miou}</strong>
            </span>
          ))}
        </div>

        {/* Viva Defense Summary Box */}
        <div className="p-2.5 rounded bg-slate-900/90 border border-cyan-500/30 text-[11px] text-slate-300 leading-relaxed flex items-start gap-2">
          <ShieldCheck className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
          <span>{viva_defense_summary}</span>
        </div>
      </div>
    </div>
  );
}
