import React, { useState } from "react";
import Sidebar, { NAV_ITEMS } from "./Sidebar.jsx";
import Topbar from "./Topbar.jsx";

export default function AppLayout({
  activePage,
  onNavigate,
  analysisData,
  health,
  isLoading,
  onOpenAOIModal,
  onOpenSitRep,
  onOpenWalkthrough,
  children,
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  const currentPageObj = NAV_ITEMS.find((item) => item.id === activePage) || NAV_ITEMS[0];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex overflow-x-hidden font-sans antialiased">
      {/* Sidebar Navigation */}
      <Sidebar
        activePage={activePage}
        onNavigate={onNavigate}
        isCollapsed={isCollapsed}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
        isMobileOpen={isMobileOpen}
        onCloseMobile={() => setIsMobileOpen(false)}
        onOpenWalkthrough={onOpenWalkthrough}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 min-h-screen overflow-y-auto">
        {/* Topbar */}
        <Topbar
          activePage={activePage}
          pageTitle={currentPageObj.label}
          analysisData={analysisData}
          health={health}
          isLoading={isLoading}
          onOpenMobileMenu={() => setIsMobileOpen(true)}
          onOpenAOIModal={onOpenAOIModal}
          onOpenSitRep={onOpenSitRep}
          onOpenWalkthrough={onOpenWalkthrough}
        />

        {/* Page Content View */}
        <main className="flex-1 pb-12">
          {children}
        </main>
      </div>
    </div>
  );
}

