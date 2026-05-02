import { useEffect, useRef, useState } from "react";
import { Link } from "@tanstack/react-router";
import {
  Wifi,
  WifiOff,
  ChevronDown,
  LogOut,
  User as UserIcon,
  Download,
  CloudOff,
  Radio,
} from "lucide-react";
import { useSensorData, formatRelative } from "@/lib/resilience";
import { NotificationBell } from "@/components/NotificationBell";

export function TopHeader() {
  const { isLive, lastSyncTs } = useSensorData();

  const [, setNow] = useState(0);
  const [online, setOnline] = useState(true);
  const [profileOpen, setProfileOpen] = useState(false);
  const [installed, setInstalled] = useState(false);

  const [user, setUser] = useState<any>(null);

  const [sweep, setSweep] = useState(false);
  const lastSeenSyncRef = useRef<number | null>(null);

  // ⏱ Keep relative time fresh
  useEffect(() => {
    const t = setInterval(() => setNow((n) => (n + 1) % 1_000_000), 1000);
    return () => clearInterval(t);
  }, []);

  // 🌐 Online/offline detection
  useEffect(() => {
    if (typeof window === "undefined") return;
    setOnline(navigator.onLine);

    const on = () => setOnline(true);
    const off = () => setOnline(false);

    window.addEventListener("online", on);
    window.addEventListener("offline", off);

    return () => {
      window.removeEventListener("online", on);
      window.removeEventListener("offline", off);
    };
  }, []);

  // 🔄 Delta sync animation
  useEffect(() => {
    if (!isLive || !lastSyncTs) return;
    if (lastSeenSyncRef.current === lastSyncTs) return;

    lastSeenSyncRef.current = lastSyncTs;
    setSweep(true);

    const t = window.setTimeout(() => setSweep(false), 950);
    return () => window.clearTimeout(t);
  }, [isLive, lastSyncTs]);

  // 👤 Fetch logged-in user from PostgreSQL via API
  useEffect(() => {
    fetch("/api/me")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.user) setUser(data.user);
      })
      .catch(() => setUser(null));
  }, []);

  const relSync = formatRelative(lastSyncTs);

  return (
    <header className="sticky top-0 z-20 -mx-4 sm:-mx-6 lg:-mx-8 mb-4 px-4 sm:px-6 lg:px-8 py-3 backdrop-blur-md bg-background/60 border-b border-border">
      <div className="flex items-center gap-3 flex-wrap">
        
        {/* Live / Historical Status */}
        {isLive ? (
          <div
            className="delta-sweep inline-flex items-center gap-2 rounded-full border border-clean/30 bg-clean/10 px-3 py-1.5 text-xs"
            data-sweep={sweep ? "on" : "off"}
            title={`Delta-sync · last fetch ${relSync}`}
          >
            <span className="delta-ring">
              <span className="live-dot block" />
            </span>
            <span className="text-clean font-medium">Live Sync</span>
            <span className="text-muted-foreground">· Δ {relSync}</span>
          </div>
        ) : (
          <div className="inline-flex items-center gap-2 rounded-full border border-moderate/40 bg-moderate/10 px-3 py-1.5 text-xs">
            <CloudOff className="h-3.5 w-3.5 text-moderate" />
            <span className="text-moderate font-medium">Historical Mode</span>
            <span className="text-muted-foreground">· {relSync}</span>
          </div>
        )}

        {/* Network Status */}
        <div
          className={[
            "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1.5 text-xs",
            online
              ? "border-cyan/30 bg-cyan/10 text-cyan"
              : "border-poor/40 bg-poor/10 text-poor",
          ].join(" ")}
        >
          {online ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
          <span className="hidden sm:inline">{online ? "Online" : "Offline"}</span>
        </div>

        {/* PWA Install */}
        <button
          onClick={() => setInstalled(true)}
          className="hidden sm:inline-flex items-center gap-1.5 rounded-full border border-border bg-panel px-2.5 py-1.5 text-xs text-muted-foreground hover:text-foreground"
        >
          <Download className="h-3.5 w-3.5" />
          {installed ? "PWA Ready" : "PWA Installable"}
        </button>

        <div className="ml-auto flex items-center gap-3">
          <NotificationBell />

          {/* Profile Dropdown */}
          <div className="relative">
            <button
              onClick={() => setProfileOpen((v) => !v)}
              className="inline-flex items-center gap-2 rounded-full border border-border bg-panel pl-1 pr-2.5 py-1 text-xs"
            >
              <span className="grid place-items-center h-7 w-7 rounded-full bg-clean/15 text-clean overflow-hidden">
                {user?.avatar ? (
                  <img
                    src={user.avatar}
                    referrerPolicy="no-referrer"
                    className="h-full w-full object-cover"
                    alt="Profile"
                  />
                ) : (
                  <UserIcon className="h-3.5 w-3.5" />
                )}
              </span>

              <span className="hidden sm:inline font-medium">
                {user?.name || "Operator"}
              </span>
              <ChevronDown className={`h-3 w-3 text-muted-foreground transition-transform ${profileOpen ? 'rotate-180' : ''}`} />
            </button>

            {profileOpen && (
              <>
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setProfileOpen(false)}
                />
                <div className="absolute right-0 mt-2 w-56 panel p-2 z-20 shadow-xl border border-border bg-background">
                  <div className="px-2 py-2 text-xs">
                    <div className="font-medium">{user?.name || "Authorized Personnel"}</div>
                    <div className="text-muted-foreground truncate">
                      {user?.email || "system-operator@envirosense.ai"}
                    </div>
                  </div>

                  <div className="my-1 border-t border-border" />

                  <button
                    onClick={() => (window.location.href = "/logout")}
                    className="w-full flex items-center gap-2 rounded-md px-2 py-1.5 text-xs hover:bg-poor/10 text-poor transition-colors"
                  >
                    <LogOut className="h-3.5 w-3.5" /> Sign out
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
