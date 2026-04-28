import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Bell, Save, Sliders } from "lucide-react";
import { Layout } from "@/components/Layout";
import { PageHeader } from "@/components/PageHeader";
import {
  readNotificationPrefs,
  writeNotificationPrefs,
  type NotificationPrefs,
} from "@/components/NotificationBell";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Settings · EnviroSense AI" },
      {
        name: "description",
        content:
          "Configure pollutant thresholds, alert rules and the polling interval that drives micro-batched updates.",
      },
      { property: "og:title", content: "Settings · EnviroSense AI" },
      { property: "og:description", content: "Thresholds, alert rules & micro-batching." },
    ],
  }),
  component: SettingsPage,
});

function SettingsPage() {
  const [clean, setClean] = useState(15);
  const [moderate, setModerate] = useState(40);
  const [interval, setIntervalSec] = useState(60);
  const [enableAdvisory, setEnableAdvisory] = useState(true);
  const [enableDrift, setEnableDrift] = useState(true);
  const [enableRegime, setEnableRegime] = useState(false);
  const [saved, setSaved] = useState(false);

  // Alert Preferences — synced with the bell icon in the global header.
  const [notifPrefs, setNotifPrefs] = useState<NotificationPrefs>({
    enabled: false,
    minSeverity: "warning",
  });
  useEffect(() => {
    setNotifPrefs(readNotificationPrefs());
    const onSync = (e: Event) => {
      const ce = e as CustomEvent<NotificationPrefs>;
      if (ce.detail) setNotifPrefs(ce.detail);
    };
    window.addEventListener("envirosense:notif-prefs", onSync);
    return () => window.removeEventListener("envirosense:notif-prefs", onSync);
  }, []);
  const updateNotifPrefs = (patch: Partial<NotificationPrefs>) => {
    const next = { ...notifPrefs, ...patch };
    setNotifPrefs(next);
    writeNotificationPrefs(next);
  };


  return (
    <Layout>
      <PageHeader
        eyebrow="Configuration"
        title="Settings"
        description="Tune thresholds, alert rules and how aggressively the console polls the FastAPI backend."
      />

      <form
        onSubmit={(e) => {
          e.preventDefault();
          setSaved(true);
          setTimeout(() => setSaved(false), 2500);
        }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-4"
      >
        <section className="panel p-5">
          <div className="flex items-center gap-2">
            <Sliders className="h-4 w-4 text-clean" />
            <div className="text-sm font-medium">Pollutant Thresholds</div>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            Boundaries between Clean / Moderate / Poor air clusters (PM2.5, µg/m³).
          </p>

          <div className="mt-4 space-y-5">
            <SliderRow
              label="Clean ↔ Moderate"
              value={clean}
              min={5}
              max={30}
              tone="clean"
              onChange={setClean}
            />
            <SliderRow
              label="Moderate ↔ Poor"
              value={moderate}
              min={25}
              max={80}
              tone="moderate"
              onChange={setModerate}
            />
          </div>

          <div className="mt-4 grid grid-cols-3 gap-2 text-center text-[11px]">
            <Bucket label="Clean" range={`< ${clean}`} tone="clean" />
            <Bucket label="Moderate" range={`${clean}–${moderate}`} tone="moderate" />
            <Bucket label="Poor" range={`> ${moderate}`} tone="poor" />
          </div>
        </section>

        <section className="panel p-5">
          <div className="text-sm font-medium">Micro-Batching</div>
          <p className="mt-1 text-xs text-muted-foreground">
            Polling interval to the FastAPI backend. Lower = fresher data, higher network/battery cost.
          </p>
          <div className="mt-4">
            <SliderRow
              label="Polling interval"
              value={interval}
              min={15}
              max={300}
              step={15}
              tone="clean"
              suffix="s"
              onChange={setIntervalSec}
            />
          </div>

          <div className="mt-6 text-sm font-medium">Alert Rules</div>
          <p className="mt-1 text-xs text-muted-foreground">
            Toggle which AI signals fire alerts in the action center.
          </p>
          <div className="mt-3 space-y-2">
            <Toggle label="AI Advisory recommendations" value={enableAdvisory} onChange={setEnableAdvisory} />
            <Toggle label="Sensor drift detection" value={enableDrift} onChange={setEnableDrift} />
            <Toggle label="Regime transition alerts" value={enableRegime} onChange={setEnableRegime} />
          </div>
        </section>

        {/* Alert Preferences — push notification opt-in */}
        <section className="panel p-5 lg:col-span-2">
          <div className="flex items-center gap-2">
            <Bell className="h-4 w-4 text-clean" />
            <div className="text-sm font-medium">Alert Preferences</div>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            Control OS-level push notifications and the minimum severity that triggers them.
          </p>

          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
            <Toggle
              label="Push Notifications (OS Level)"
              value={notifPrefs.enabled}
              onChange={(v) => updateNotifPrefs({ enabled: v })}
            />
            <label className="flex items-center justify-between gap-3 rounded-md border border-border bg-panel/50 px-3 py-2 text-xs">
              <span>Minimum severity to push</span>
              <select
                value={notifPrefs.minSeverity}
                onChange={(e) =>
                  updateNotifPrefs({
                    minSeverity: e.target.value as NotificationPrefs["minSeverity"],
                  })
                }
                disabled={!notifPrefs.enabled}
                className="rounded-md border border-border bg-panel px-2 py-1 text-xs text-foreground focus:outline-none focus:border-clean/50 disabled:opacity-50 [color-scheme:dark]"
              >
                <option value="info">Info</option>
                <option value="warning">Warning</option>
                <option value="critical">Critical</option>
              </select>
            </label>
          </div>

          <div className="mt-3 text-[11px] text-muted-foreground">
            {notifPrefs.enabled
              ? `Push enabled · alerts at "${notifPrefs.minSeverity}" or higher will be sent to your device.`
              : "Push notifications are off. Enable above or click the bell icon in the header."}
          </div>
        </section>

        <div className="lg:col-span-2 flex items-center justify-end gap-3">
          {saved && <span className="text-xs text-clean">Settings saved (locally).</span>}
          <button
            type="submit"
            className="inline-flex items-center gap-2 rounded-lg bg-clean text-primary-foreground px-4 py-2 text-sm font-semibold shadow-[0_0_20px_oklch(0.78_0.18_150_/_0.3)] hover:bg-clean/90 transition-colors"
          >
            <Save className="h-4 w-4" /> Save changes
          </button>
        </div>
      </form>
    </Layout>
  );
}

function SliderRow({
  label,
  value,
  min,
  max,
  step = 1,
  tone,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  tone: "clean" | "moderate" | "poor";
  suffix?: string;
  onChange: (v: number) => void;
}) {
  const colorMap = { clean: "accent-clean text-clean", moderate: "accent-moderate text-moderate", poor: "accent-poor text-poor" };
  return (
    <div>
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className={`font-mono font-medium ${colorMap[tone]}`}>
          {value}{suffix ?? ""}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        className={`mt-1.5 w-full ${colorMap[tone]}`}
      />
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>{min}{suffix ?? ""}</span>
        <span>{max}{suffix ?? ""}</span>
      </div>
    </div>
  );
}

function Bucket({ label, range, tone }: { label: string; range: string; tone: "clean" | "moderate" | "poor" }) {
  const colorMap = {
    clean: "border-clean/40 bg-clean/10 text-clean",
    moderate: "border-moderate/40 bg-moderate/10 text-moderate",
    poor: "border-poor/40 bg-poor/10 text-poor",
  } as const;
  return (
    <div className={`rounded-lg border px-2 py-2 ${colorMap[tone]}`}>
      <div className="text-[10px] uppercase tracking-wider">{label}</div>
      <div className="font-mono text-xs">{range}</div>
    </div>
  );
}

function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center justify-between gap-3 rounded-md border border-border bg-panel/50 px-3 py-2 text-xs cursor-pointer">
      <span>{label}</span>
      <span
        className={`relative h-5 w-9 rounded-full border transition-colors ${value ? "bg-clean/30 border-clean/50" : "bg-panel border-border"}`}
        onClick={() => onChange(!value)}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full transition-transform ${value ? "left-0.5 translate-x-4 bg-clean" : "left-0.5 bg-muted-foreground"}`}
        />
      </span>
      <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} className="sr-only" />
    </label>
  );
}
