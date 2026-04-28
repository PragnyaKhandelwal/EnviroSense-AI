import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, BellRing, Check, CheckCircle2, Info, Lightbulb, Sparkles, ShieldAlert, X } from "lucide-react";
import { Layout } from "@/components/Layout";
import { PageHeader } from "@/components/PageHeader";
import type { AlertSeverity, AlertStatus, AppAlert } from "@/lib/mock-data";
import { useSensorData } from "@/lib/resilience";
import { HistoricalBanner } from "@/components/HistoricalBanner";

export const Route = createFileRoute("/alerts")({
  head: () => ({
    meta: [
      { title: "Alerts & Actions · EnviroSense AI" },
      {
        name: "description",
        content:
          "Action center: prioritised alert feed with AI advisories and lifecycle controls (Acknowledge / Resolve).",
      },
      { property: "og:title", content: "Alerts & Actions · EnviroSense AI" },
      { property: "og:description", content: "Anomaly action center with AI advisories." },
    ],
  }),
  component: AlertsPage,
});

const SEVERITIES: Array<AlertSeverity | "all"> = ["all", "critical", "warning", "info"];
const STATUSES: Array<AlertStatus | "all"> = ["all", "open", "acknowledged", "resolved"];

function AlertsPage() {
  const { isLive, snapshot } = useSensorData();
  // Seed alert list from the live pipeline snapshot. Local state is then used
  // for user-driven lifecycle changes (acknowledge / resolve) so we do not
  // overwrite their actions when the snapshot refreshes.
  const [alerts, setAlerts] = useState<AppAlert[]>(snapshot.alerts);
  const [sev, setSev] = useState<AlertSeverity | "all">("all");
  const [status, setStatus] = useState<AlertStatus | "all">("all");

  // First open critical alert powers the toast — no longer hardcoded.
  const firstCritical = useMemo(
    () => snapshot.alerts.find((a) => a.severity === "critical" && a.status !== "resolved"),
    [snapshot.alerts],
  );
  const [toast, setToast] = useState<string | null>(
    firstCritical ? `New critical alert: ${firstCritical.title} · ${firstCritical.id}` : null,
  );

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 6000);
    return () => clearTimeout(t);
  }, [toast]);

  const filtered = useMemo(
    () =>
      alerts.filter(
        (a) => (sev === "all" || a.severity === sev) && (status === "all" || a.status === status),
      ),
    [alerts, sev, status],
  );

  const counts = useMemo(
    () => ({
      critical: alerts.filter((a) => a.severity === "critical" && a.status !== "resolved").length,
      warning: alerts.filter((a) => a.severity === "warning" && a.status !== "resolved").length,
      info: alerts.filter((a) => a.severity === "info" && a.status !== "resolved").length,
    }),
    [alerts],
  );

  const setStatusOf = (id: string, next: AlertStatus) =>
    setAlerts((prev) => prev.map((a) => (a.id === id ? { ...a, status: next } : a)));

  return (
    <Layout>
      <PageHeader
        eyebrow="Action Center"
        title="Alerts & Solutions"
        description="Triage anomalies, follow AI recommendations, and manage the lifecycle of every alert."
      />

      <HistoricalBanner message="Live inference paused. Showing cached alerts. Acknowledge / Resolve actions disabled while offline." />

      {toast && (
        <div className="mb-4 panel panel-glow-poor p-3 flex items-start gap-3">
          <AlertTriangle className="h-4 w-4 text-poor mt-0.5" />
          <div className="flex-1 text-sm">
            <div className="font-medium text-poor">Anomaly detected</div>
            <div className="text-xs text-muted-foreground">{toast}</div>
          </div>
          <button onClick={() => setToast(null)} className="text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        <SummaryCard tone="poor" icon={ShieldAlert} label="Critical · Open" value={counts.critical} />
        <SummaryCard tone="moderate" icon={AlertTriangle} label="Warnings · Open" value={counts.warning} />
        <SummaryCard tone="cyan" icon={Info} label="Info · Open" value={counts.info} />
      </div>

      <div className="panel p-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Alert Feed</div>
            <div className="mt-0.5 text-sm flex items-center gap-2">
              <BellRing className="h-3.5 w-3.5 text-moderate" />
              {filtered.length} {filtered.length === 1 ? "alert" : "alerts"}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <FilterPills value={sev} onChange={setSev} options={SEVERITIES} />
            <FilterPills value={status} onChange={setStatus} options={STATUSES} />
          </div>
        </div>

        <ul className="mt-4 space-y-3">
          {filtered.map((a) => (
            <AlertItem key={a.id} alert={a} onSetStatus={setStatusOf} isLive={isLive} />
          ))}
          {filtered.length === 0 && (
            <li className="text-sm text-muted-foreground text-center py-8">
              No alerts match the current filters.
            </li>
          )}
        </ul>
      </div>
    </Layout>
  );
}

function SummaryCard({
  tone,
  icon: Icon,
  label,
  value,
}: {
  tone: "poor" | "moderate" | "cyan";
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: number;
}) {
  const colorMap = {
    poor: "text-poor border-poor/40 bg-poor/10",
    moderate: "text-moderate border-moderate/40 bg-moderate/10",
    cyan: "text-cyan border-cyan/40 bg-cyan/10",
  } as const;
  return (
    <div className={`panel p-4 flex items-center gap-3 border ${colorMap[tone]}`}>
      <Icon className="h-5 w-5" />
      <div>
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
        <div className="font-mono text-2xl font-semibold">{value}</div>
      </div>
    </div>
  );
}

function FilterPills<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: readonly T[];
}) {
  return (
    <div className="inline-flex rounded-lg border border-border bg-panel p-0.5 text-xs">
      {options.map((o) => (
        <button
          key={o}
          onClick={() => onChange(o)}
          className={`px-2.5 py-1 rounded-md transition-colors capitalize ${
            value === o ? "bg-clean/15 text-clean" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {o}
        </button>
      ))}
    </div>
  );
}

type FeedbackVerdict = "confirmed" | "false_alarm";
type FeedbackContext = "Cooking" | "Window Opened" | "Weather" | "Other";

interface AlertFeedback {
  verdict: FeedbackVerdict;
  context?: FeedbackContext;
  ts: number;
}

const FEEDBACK_KEY = "envirosense.feedback.v1";

function readFeedback(): Record<string, AlertFeedback> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(FEEDBACK_KEY);
    if (!raw) return {};
    return JSON.parse(raw) as Record<string, AlertFeedback>;
  } catch {
    return {};
  }
}

function writeFeedback(map: Record<string, AlertFeedback>) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(FEEDBACK_KEY, JSON.stringify(map));
  } catch {
    /* ignore */
  }
}

const FEEDBACK_CONTEXTS: FeedbackContext[] = ["Cooking", "Window Opened", "Weather", "Other"];

function AlertItem({
  alert,
  onSetStatus,
  isLive,
}: {
  alert: AppAlert;
  onSetStatus: (id: string, status: AlertStatus) => void;
  isLive: boolean;
}) {
  const sevMap = {
    critical: { color: "text-poor border-poor/40 bg-poor/10", panel: "panel-glow-poor", icon: ShieldAlert },
    warning: { color: "text-moderate border-moderate/40 bg-moderate/10", panel: "", icon: AlertTriangle },
    info: { color: "text-cyan border-cyan/40 bg-cyan/10", panel: "", icon: Info },
  } as const;
  const statusMap = {
    open: "text-poor",
    acknowledged: "text-moderate",
    resolved: "text-clean",
  } as const;
  const s = sevMap[alert.severity];
  const Icon = s.icon;

  // Local feedback state, persisted per-alert in localStorage.
  const [feedback, setFeedback] = useState<AlertFeedback | null>(null);
  const [pendingConfirm, setPendingConfirm] = useState(false);

  useEffect(() => {
    const map = readFeedback();
    if (map[alert.id]) setFeedback(map[alert.id]);
  }, [alert.id]);

  const recordFeedback = (next: AlertFeedback) => {
    setFeedback(next);
    setPendingConfirm(false);
    const map = readFeedback();
    map[alert.id] = next;
    writeFeedback(map);
  };

  return (
    <li className={`panel ${alert.severity === "critical" ? s.panel : ""} p-4`}>
      <div className="flex items-start gap-3">
        <span className={`grid place-items-center h-9 w-9 rounded-lg border ${s.color} shrink-0`}>
          <Icon className="h-4 w-4" />
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className={`text-[10px] uppercase tracking-wider rounded px-1.5 py-0.5 border ${s.color}`}>
              {alert.severity}
            </span>
            <span className={`text-[10px] uppercase tracking-wider font-medium ${statusMap[alert.status]}`}>
              {alert.status}
            </span>
            <span className="text-[10px] text-muted-foreground font-mono">{alert.id}</span>
            <span className="text-[10px] text-muted-foreground ml-auto font-mono">{alert.ts}</span>
          </div>
          <div className="mt-1 text-sm font-medium">{alert.title}</div>
          <div className="text-xs text-muted-foreground">{alert.description}</div>

          {alert.recommendation && (
            <div className="mt-3 rounded-lg border border-clean/30 bg-clean/5 p-3 flex items-start gap-2">
              <Lightbulb className="h-3.5 w-3.5 text-clean shrink-0 mt-0.5" />
              <div className="text-xs">
                <div className="text-[10px] uppercase tracking-wider text-clean font-semibold">AI Advisory</div>
                <div className="mt-0.5 text-foreground/90">{alert.recommendation}</div>
              </div>
            </div>
          )}

          {alert.status !== "resolved" && (
            <div className="mt-3 flex flex-wrap gap-2">
              {alert.status === "open" && (
                <button
                  onClick={() => isLive && onSetStatus(alert.id, "acknowledged")}
                  disabled={!isLive}
                  title={isLive ? "Acknowledge alert" : "Action disabled while offline."}
                  className="inline-flex items-center gap-1.5 rounded-md border border-moderate/40 bg-moderate/10 text-moderate px-2.5 py-1 text-xs font-medium hover:bg-moderate/15 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-moderate/10"
                >
                  <CheckCircle2 className="h-3.5 w-3.5" /> Acknowledge
                </button>
              )}
              <button
                onClick={() => isLive && onSetStatus(alert.id, "resolved")}
                disabled={!isLive}
                title={isLive ? "Mark as resolved" : "Action disabled while offline."}
                className="inline-flex items-center gap-1.5 rounded-md border border-clean/40 bg-clean/10 text-clean px-2.5 py-1 text-xs font-medium hover:bg-clean/15 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-clean/10"
              >
                <CheckCircle2 className="h-3.5 w-3.5" /> Resolve
              </button>
            </div>
          )}

          {/* AI Feedback bar — Human-in-the-loop */}
          <div className="mt-3 pt-3 border-t border-border/60">
            {feedback ? (
              <div className="flex items-center gap-2 rounded-md border border-clean/30 bg-clean/5 px-2.5 py-1.5 text-xs text-clean">
                <Sparkles className="h-3.5 w-3.5" />
                <span className="font-medium">Feedback logged</span>
                <span className="text-muted-foreground">
                  · {feedback.verdict === "confirmed" ? "Confirmed event" : "Marked as false alarm"}
                  {feedback.context ? ` · ${feedback.context}` : ""}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    setFeedback(null);
                    const map = readFeedback();
                    delete map[alert.id];
                    writeFeedback(map);
                  }}
                  className="ml-auto text-[10px] text-muted-foreground hover:text-foreground"
                >
                  Undo
                </button>
              </div>
            ) : pendingConfirm ? (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  What caused this event?
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {FEEDBACK_CONTEXTS.map((c) => (
                    <button
                      key={c}
                      type="button"
                      onClick={() =>
                        recordFeedback({ verdict: "confirmed", context: c, ts: Date.now() })
                      }
                      className="rounded-full border border-clean/40 bg-clean/10 text-clean px-2.5 py-1 text-[11px] font-medium hover:bg-clean/20 transition-colors"
                    >
                      {c}
                    </button>
                  ))}
                  <button
                    type="button"
                    onClick={() => setPendingConfirm(false)}
                    className="rounded-full border border-border bg-panel px-2.5 py-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  AI Feedback
                </span>
                <button
                  type="button"
                  onClick={() => setPendingConfirm(true)}
                  className="inline-flex items-center gap-1.5 rounded-md border border-clean/40 bg-clean/10 text-clean px-2.5 py-1 text-xs font-medium hover:bg-clean/15 transition-colors"
                >
                  <Check className="h-3.5 w-3.5" /> Confirm Event
                </button>
                <button
                  type="button"
                  onClick={() =>
                    recordFeedback({ verdict: "false_alarm", ts: Date.now() })
                  }
                  className="inline-flex items-center gap-1.5 rounded-md border border-poor/30 bg-poor/5 text-poor/90 px-2.5 py-1 text-xs font-medium hover:bg-poor/10 transition-colors"
                >
                  <X className="h-3.5 w-3.5" /> False Alarm
                </button>
                <span className="ml-auto text-[10px] text-muted-foreground hidden sm:inline">
                  Helps train the model
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </li>
  );
}
