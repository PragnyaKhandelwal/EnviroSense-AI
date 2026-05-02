import { Link, useLocation } from "@tanstack/react-router";
import {
  LayoutDashboard,
  Brain,
  BellRing,
  ShieldCheck,
  Settings as SettingsIcon,
  Menu,
  X,
  Wind,
} from "lucide-react";
import { useState } from "react";

const items = [
  { to: "/", label: "Overview", icon: LayoutDashboard, exact: true, description: "Live dashboard" },
  { to: "/predictive", label: "Predictive Intelligence", icon: Brain, description: "Forecasts & regimes" },
  { to: "/alerts", label: "Alerts & Actions", icon: BellRing, description: "Action center" },
  { to: "/system-health", label: "Data Trust & Health", icon: ShieldCheck, description: "Quality & drift" },
  { to: "/settings", label: "Settings", icon: SettingsIcon, description: "Thresholds & rules" },
] as const;

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { pathname } = useLocation();

  const isActive = (to: string, exact?: boolean) =>
    exact ? pathname === to : pathname === to || pathname.startsWith(to + "/");

  return (
    <>
      {/* Mobile top bar */}
      <div className="md:hidden sticky top-0 z-30 flex items-center justify-between px-4 py-3 panel rounded-none border-x-0 border-t-0">
        <div className="flex items-center gap-2">
          <Wind className="h-5 w-5 text-clean glow-text-clean" />
          <span className="font-semibold tracking-tight">EnviroSense AI</span>
        </div>
        <button
          aria-label="Open menu"
          onClick={() => setMobileOpen((v) => !v)}
          className="rounded-md border border-border p-2 hover:bg-secondary"
        >
          {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </button>
      </div>

      {/* Sidebar */}
      <aside
        className={[
          "panel rounded-none md:rounded-r-xl border-l-0",
          "md:sticky md:top-0 md:h-screen md:flex flex-col",
          "transition-[width] duration-300 ease-out",
          collapsed ? "md:w-[72px]" : "md:w-[260px]",
          mobileOpen ? "block" : "hidden md:flex",
          "z-20",
        ].join(" ")}
      >
        <div className="hidden md:flex items-center justify-between px-4 py-5">
          <div
            className={[
              "flex items-center gap-2 overflow-hidden",
              collapsed ? "justify-center w-full" : "",
            ].join(" ")}
          >
            <div className="grid place-items-center h-9 w-9 rounded-lg bg-clean/15 border border-clean/30">
              <Wind className="h-5 w-5 text-clean glow-text-clean" />
            </div>
            {!collapsed && (
              <div className="leading-tight">
                <div className="font-semibold tracking-tight">EnviroSense</div>
                <div className="text-xs text-muted-foreground">AI Console</div>
              </div>
            )}
          </div>
          {!collapsed && (
            <button
              aria-label="Collapse sidebar"
              onClick={() => setCollapsed(true)}
              className="rounded-md border border-border p-1.5 hover:bg-secondary"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        {collapsed && (
          <button
            aria-label="Expand sidebar"
            onClick={() => setCollapsed(false)}
            className="hidden md:flex mx-auto mb-2 rounded-md border border-border p-1.5 hover:bg-secondary"
          >
            <Menu className="h-3.5 w-3.5" />
          </button>
        )}

        <nav className="flex-1 px-2 py-2 md:py-0 space-y-1">
          {items.map((it) => {
            const Active = isActive(it.to, "exact" in it ? it.exact : false);
            const Icon = it.icon;
            return (
              <Link
                key={it.to}
                to={it.to}
                onClick={() => setMobileOpen(false)}
                className={[
                  "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
                  Active
                    ? "bg-clean/10 text-clean border border-clean/30"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary border border-transparent",
                ].join(" ")}
              >
                <Icon className={["h-4 w-4 shrink-0", Active ? "glow-text-clean" : ""].join(" ")} />
                {!collapsed && (
                  <div className="min-w-0 flex-1">
                    <div className="truncate">{it.label}</div>
                    <div className="text-[10px] text-muted-foreground truncate">{it.description}</div>
                  </div>
                )}
                {Active && !collapsed && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-clean" />}
              </Link>
            );
          })}
        </nav>

        <div className="hidden md:block px-4 py-4 text-xs text-muted-foreground border-t border-border">
          {!collapsed ? (
            <>
              <div className="flex items-center gap-2">
                <span className="live-dot" />
                <span>Node online · v0.4.2</span>
              </div>
              <div className="mt-1 text-[11px]">My Terrace · Hyper-local node</div>
            </>
          ) : (
            <span className="live-dot mx-auto block" />
          )}
        </div>
      </aside>
    </>
  );
}
