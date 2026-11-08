import React from "react";
import {
  Activity,
  Award,
  ChevronLeft,
  ChevronRight,
  Compass,
  Database,
  FileText,
  LifeBuoy,
  MapPin,
  Navigation,
  Radar,
  Satellite,
  Settings,
  Waves,
  Wind,
} from "lucide-react";

export const NAV_ITEMS = [
  {
    id: "dashboard",
    label: "Dashboard",
    icon: Activity,
    description: "Disaster Monitoring Overview",
  },
  {
    id: "pipeline",
    label: "End-to-End Pipeline",
    icon: Radar,
    description: "Unified Workflow Orchestrator",
  },
  {
    id: "satellite-analysis",
    label: "Satellite Analysis",
    icon: Satellite,
    description: "Sentinel-1/2 Dual-Stream U-Net",
  },
  {
    id: "flood-simulation",
    label: "Flood Simulation",
    icon: Wind,
    description: "2D Hydrodynamic & Cyclone Scenarios",
  },
  {
    id: "rescue",
    label: "Rescue & Infrastructure",
    icon: Navigation,
    description: "OSM Routing & Building Damage",
  },
  {
    id: "reports",
    label: "Disaster Reports",
    icon: FileText,
    description: "NDRF Action SitRep & Exports",
  },
  {
    id: "data-sources",
    label: "Data Sources",
    icon: Database,
    description: "Copernicus, OSM, GPM & Cache",
  },
  {
    id: "settings",
    label: "Settings",
    icon: Settings,
    description: "Preferences & System Diagnostics",
  },
];

export default function Sidebar({
  activePage,
  onNavigate,
  isCollapsed,
  onToggleCollapse,
  isMobileOpen,
  onCloseMobile,
  onOpenWalkthrough,
}) {
  return (
    <>
      {/* Mobile Backdrop */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-40 lg:hidden"
          onClick={onCloseMobile}
        />
      )}

      {/* Main Sidebar Shell */}
      <aside
        className={`fixed lg:static top-0 bottom-0 left-0 z-50 bg-white border-r border-slate-200 flex flex-col transition-all duration-300 shadow-sm ${
          isCollapsed ? "w-20" : "w-64"
        } ${isMobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}`}
      >
        {/* Brand Header */}
        <div className="h-16 px-4 border-b border-slate-200 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20 shrink-0">
              <Waves className="w-5 h-5" />
            </div>
            {!isCollapsed && (
              <div className="min-w-0">
                <div className="font-bold text-sm text-slate-900 tracking-tight flex items-center gap-1.5">
                  FloodSense<span className="text-blue-600">AI</span>
                </div>
                <div className="text-[10px] text-slate-500 truncate">
                  Disaster Command System
                </div>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={onToggleCollapse}
            className="hidden lg:flex w-7 h-7 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 items-center justify-center transition"
            title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
          >
            {isCollapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <ChevronLeft className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  onNavigate(item.id);
                  if (onCloseMobile) onCloseMobile();
                }}
                title={isCollapsed ? item.label : undefined}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition group relative ${
                  isActive
                    ? "bg-blue-50 text-blue-700 font-semibold"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                }`}
              >
                <Icon
                  className={`w-5 h-5 shrink-0 transition ${
                    isActive
                      ? "text-blue-600"
                      : "text-slate-400 group-hover:text-slate-600"
                  }`}
                />
                {!isCollapsed && (
                  <span className="truncate">{item.label}</span>
                )}

                {/* Collapsed Tooltip */}
                {isCollapsed && (
                  <div className="absolute left-full ml-3 px-2.5 py-1 bg-slate-900 text-white text-xs rounded-md shadow-lg whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition z-50">
                    {item.label}
                  </div>
                )}
              </button>
            );
          })}
        </nav>

        {/* Bottom Viva Defense Card */}
        <div className="p-3 border-t border-slate-200">
          <button
            type="button"
            onClick={() => {
              if (onOpenWalkthrough) onOpenWalkthrough();
              if (onCloseMobile) onCloseMobile();
            }}
            className={`w-full flex items-center gap-2.5 p-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 text-blue-700 hover:border-blue-300 transition ${
              isCollapsed ? "justify-center" : ""
            }`}
            title="Open Project Walkthrough & Architecture Viva Guide"
          >
            <Award className="w-4 h-4 text-blue-600 shrink-0" />
            {!isCollapsed && <span className="truncate">Viva Defense Guide</span>}
          </button>
        </div>
      </aside>
    </>
  );
}

