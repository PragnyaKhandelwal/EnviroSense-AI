import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import {
  Calendar as CalendarIcon,
  Download,
  FileText,
  Loader2,
  Sparkles,
  X,
} from "lucide-react";
import { useSensorData } from "@/lib/resilience";

/**
 * Premium dark slide-out drawer for exporting a custom data + chart report.
 * Mock implementation — toggling "Export" reveals a "Generating..." state and
 * resolves after a short delay, matching the demo aesthetic.
 */
export function ReportDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { isLive } = useSensorData();
  const today = new Date().toISOString().slice(0, 10);
  const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 3600_000).toISOString().slice(0, 10);

  const [from, setFrom] = useState(sevenDaysAgo);
  const [to, setTo] = useState(today);
  const [includeRaw, setIncludeRaw] = useState(true);
  const [includePred, setIncludePred] = useState(true);
  const [includeAnom, setIncludeAnom] = useState(true);
  const [format, setFormat] = useState<"csv" | "pdf">("csv");
  const [busy, setBusy] = useState<null | "csv" | "pdf">(null);
  const [done, setDone] = useState<null | string>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  // ESC to close + scroll lock
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);

  // Reset transient state when drawer reopens
  useEffect(() => {
    if (open) {
      setBusy(null);
      setDone(null);
    }
  }, [open]);

  const handleExport = (kind: "csv" | "pdf") => {
    if (busy) return;
    setBusy(kind);
    setDone(null);
    // Mock zipping / generation pipeline
    window.setTimeout(() => {
      setBusy(null);
      setDone(
        kind === "csv"
          ? "report.csv ready · 1,284 rows packaged"
          : "report.pdf ready · 14 pages with charts",
      );
    }, 1800);
  };

  if (!mounted || !open) return null;

  return createPortal(
    <div className="fixed inset-0 z-50">
      {/* Backdrop */}
      <button
        type="button"
        aria-label="Close report drawer"
        onClick={onClose}
        className="absolute inset-0 bg-background/70 backdrop-blur-sm animate-fade-in"
      />

      {/* Drawer panel */}
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="report-drawer-title"
        className="absolute right-0 top-0 h-full w-full sm:w-[440px] panel border-l border-border bg-background/95 backdrop-blur-xl shadow-2xl flex flex-col animate-slide-in-right"
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-3 p-5 border-b border-border">
          <div>
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              Export Engine
            </div>
            <h2 id="report-drawer-title" className="mt-0.5 text-lg font-semibold flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-clean" />
              Generate Report
            </h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Bundle raw readings, AI inference and anomaly logs into a shareable artifact.
            </p>
          </div>
          <button
            onClick={onClose}
            className="grid place-items-center h-8 w-8 rounded-full border border-border text-muted-foreground hover:text-foreground hover:border-clean/40 transition-colors"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* Date range */}
          <section>
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground flex items-center gap-1.5">
              <CalendarIcon className="h-3 w-3" /> Date Range
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <label className="block">
                <span className="text-[10px] text-muted-foreground">Start</span>
                <input
                  type="date"
                  value={from}
                  max={to}
                  onChange={(e) => setFrom(e.target.value)}
                  className="mt-1 w-full rounded-md border border-border bg-panel px-2.5 py-1.5 text-xs text-foreground focus:outline-none focus:border-clean/50 [color-scheme:dark]"
                />
              </label>
              <label className="block">
                <span className="text-[10px] text-muted-foreground">End</span>
                <input
                  type="date"
                  value={to}
                  min={from}
                  max={today}
                  onChange={(e) => setTo(e.target.value)}
                  className="mt-1 w-full rounded-md border border-border bg-panel px-2.5 py-1.5 text-xs text-foreground focus:outline-none focus:border-clean/50 [color-scheme:dark]"
                />
              </label>
            </div>
          </section>

          {/* Toggles */}
          <section>
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              Include in Export
            </div>
            <div className="mt-2 space-y-2">
              <DrawerToggle label="Raw sensor data" value={includeRaw} onChange={setIncludeRaw} />
              <DrawerToggle label="AI predictions" value={includePred} onChange={setIncludePred} />
              <DrawerToggle label="Anomaly logs" value={includeAnom} onChange={setIncludeAnom} />
            </div>
          </section>

          {/* Format */}
          <section>
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              Format
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <FormatBtn
                active={format === "csv"}
                onClick={() => setFormat("csv")}
                icon={<FileText className="h-3.5 w-3.5" />}
                title="CSV"
                subtitle="Data-focused"
              />
              <FormatBtn
                active={format === "pdf"}
                onClick={() => setFormat("pdf")}
                icon={<FileText className="h-3.5 w-3.5" />}
                title="PDF"
                subtitle="Chart-focused"
              />
            </div>
          </section>

          {/* Status / disclaimer */}
          {!isLive && (
            <div className="rounded-lg border border-moderate/40 bg-moderate/10 p-3 text-xs text-moderate">
              You are offline. The export will use the most recent cached snapshot.
            </div>
          )}
          {done && (
            <div className="rounded-lg border border-clean/40 bg-clean/10 p-3 text-xs text-clean">
              {done}
            </div>
          )}
        </div>

        {/* Footer actions */}
        <div className="border-t border-border p-4 flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border bg-panel px-3 py-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={busy !== null || (!includeRaw && !includePred && !includeAnom)}
            onClick={() => handleExport(format)}
            className="inline-flex items-center gap-2 rounded-md bg-clean text-primary-foreground px-4 py-2 text-xs font-semibold shadow-[0_0_20px_oklch(0.78_0.18_150_/_0.3)] hover:bg-clean/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {busy ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                {busy === "csv" ? "Generating CSV…" : "Generating PDF…"}
              </>
            ) : (
              <>
                <Download className="h-3.5 w-3.5" />
                Export {format.toUpperCase()}
              </>
            )}
          </button>
        </div>
      </aside>
    </div>,
    document.body,
  );
}

function DrawerToggle({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-3 rounded-md border border-border bg-panel/50 px-3 py-2 text-xs cursor-pointer hover:border-clean/30 transition-colors">
      <span>{label}</span>
      <span
        className={`relative h-5 w-9 rounded-full border transition-colors ${
          value ? "bg-clean/30 border-clean/50" : "bg-panel border-border"
        }`}
        onClick={(e) => {
          e.preventDefault();
          onChange(!value);
        }}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full transition-transform ${
            value ? "left-0.5 translate-x-4 bg-clean" : "left-0.5 bg-muted-foreground"
          }`}
        />
      </span>
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
        className="sr-only"
      />
    </label>
  );
}

function FormatBtn({
  active,
  onClick,
  icon,
  title,
  subtitle,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  title: string;
  subtitle: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`text-left rounded-lg border px-3 py-2.5 transition-colors ${
        active
          ? "border-clean/50 bg-clean/10 text-clean"
          : "border-border bg-panel text-muted-foreground hover:text-foreground hover:border-clean/30"
      }`}
    >
      <div className="flex items-center gap-1.5 text-xs font-semibold">
        {icon} {title}
      </div>
      <div className="text-[10px] text-muted-foreground mt-0.5">{subtitle}</div>
    </button>
  );
}

/**
 * Reusable trigger button for page headers — opens the drawer.
 */
export function GenerateReportButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-1.5 rounded-md border border-clean/40 bg-clean/10 text-clean px-3 py-1.5 text-xs font-semibold hover:bg-clean/15 transition-colors shadow-[0_0_18px_oklch(0.78_0.18_150_/_0.18)]"
    >
      <Download className="h-3.5 w-3.5" /> Generate Report
    </button>
  );
}
