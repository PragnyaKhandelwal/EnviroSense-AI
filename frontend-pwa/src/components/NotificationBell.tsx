import { useEffect, useState } from "react";
import { Bell, BellOff, BellRing } from "lucide-react";

const STORAGE_KEY = "envirosense.notifications.v1";

export interface NotificationPrefs {
  enabled: boolean;
  minSeverity: "info" | "warning" | "critical";
}

const DEFAULT: NotificationPrefs = { enabled: false, minSeverity: "warning" };

export function readNotificationPrefs(): NotificationPrefs {
  if (typeof window === "undefined") return DEFAULT;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT;
    const parsed = JSON.parse(raw) as NotificationPrefs;
    return {
      enabled: !!parsed.enabled,
      minSeverity:
        parsed.minSeverity === "info" ||
        parsed.minSeverity === "warning" ||
        parsed.minSeverity === "critical"
          ? parsed.minSeverity
          : "warning",
    };
  } catch {
    return DEFAULT;
  }
}

export function writeNotificationPrefs(p: NotificationPrefs) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
    // Cross-component sync within the same tab.
    window.dispatchEvent(new CustomEvent("envirosense:notif-prefs", { detail: p }));
  } catch {
    /* ignore */
  }
}

/**
 * Bell icon for the global header. Mock-only: tracks an "enabled" flag in
 * localStorage and shows a pulsing badge when the user has not opted in yet.
 * Clicking flips the flag and broadcasts a custom event so the Settings page
 * stays in sync without a page reload.
 */
export function NotificationBell() {
  const [prefs, setPrefs] = useState<NotificationPrefs>(DEFAULT);
  const [tipOpen, setTipOpen] = useState(false);

  // Hydrate from localStorage after mount (SSR-safe).
  useEffect(() => {
    setPrefs(readNotificationPrefs());
    const onSync = (e: Event) => {
      const ce = e as CustomEvent<NotificationPrefs>;
      if (ce.detail) setPrefs(ce.detail);
    };
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setPrefs(readNotificationPrefs());
    };
    window.addEventListener("envirosense:notif-prefs", onSync);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener("envirosense:notif-prefs", onSync);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  const enabled = prefs.enabled;

  const Icon = enabled ? BellRing : Bell;
  const labelTone = enabled ? "text-clean" : "text-muted-foreground";

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => {
          const next = { ...prefs, enabled: !enabled };
          setPrefs(next);
          writeNotificationPrefs(next);
          setTipOpen(false);
        }}
        onMouseEnter={() => !enabled && setTipOpen(true)}
        onMouseLeave={() => setTipOpen(false)}
        title={enabled ? "Push notifications enabled" : "Enable critical alerts"}
        aria-label={enabled ? "Disable push notifications" : "Enable push notifications"}
        className={[
          "relative inline-flex items-center justify-center h-8 w-8 rounded-full border bg-panel transition-colors",
          enabled
            ? "border-clean/40 text-clean hover:border-clean/60"
            : "border-border text-muted-foreground hover:text-foreground hover:border-clean/40",
        ].join(" ")}
      >
        <Icon className="h-3.5 w-3.5" />
        {!enabled && (
          <span
            aria-hidden
            className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-poor delta-ring"
          />
        )}
        {enabled && (
          <BellOff className="hidden" /> /* keep tree-shaker alive in case of future swap */
        )}
      </button>

      {/* Tooltip prompt when disabled */}
      {tipOpen && !enabled && (
        <div className="absolute right-0 top-full mt-2 z-30 w-56 panel p-2.5 text-xs">
          <div className={`font-medium ${labelTone}`}>Notifications off</div>
          <div className="text-muted-foreground mt-0.5">
            Click the bell to enable critical alerts at the OS level.
          </div>
        </div>
      )}
    </div>
  );
}
