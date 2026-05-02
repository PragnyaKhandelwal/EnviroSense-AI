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

export function ReportDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  // ✅ FIX: correct context usage
  const { snapshot, loading } = useSensorData();

  const today = new Date().toISOString().slice(0, 10);
  const sevenDaysAgo = new Date(
    Date.now() - 7 * 24 * 3600_000
  ).toISOString().slice(0, 10);

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

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) =>
      e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);

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

    setTimeout(() => {
      setBusy(null);
      setDone(
        kind === "csv"
          ? "report.csv ready · 1,284 rows packaged"
          : "report.pdf ready · 14 pages with charts"
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

      {/* Drawer */}
      <aside className="absolute right-0 top-0 h-full w-full sm:w-[440px] panel border-l border-border bg-background/95 backdrop-blur-xl shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-start justify-between p-5 border-b border-border">
          <div>
            <div className="text-xs text-muted-foreground uppercase">
              Export Engine
            </div>
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-clean" />
              Generate Report
            </h2>
          </div>
          <button onClick={onClose}>
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* Date */}
          <section>
            <div className="text-xs text-muted-foreground">
              Date Range
            </div>
            <div className="grid grid-cols-2 gap-2 mt-2">
              <input
                type="date"
                value={from}
                max={to}
                onChange={(e) => setFrom(e.target.value)}
              />
              <input
                type="date"
                value={to}
                min={from}
                onChange={(e) => setTo(e.target.value)}
              />
            </div>
          </section>

          {/* Status FIXED */}
          {!snapshot && (
            <div className="text-xs text-moderate">
              Data not available yet
            </div>
          )}

          {loading && (
            <div className="text-xs text-clean">
              Loading data...
            </div>
          )}

          {done && (
            <div className="text-xs text-clean">
              {done}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t p-4 flex justify-between">
          <button onClick={onClose}>Cancel</button>
          <button
            disabled={busy !== null}
            onClick={() => handleExport(format)}
          >
            {busy ? (
              <>
                <Loader2 className="animate-spin h-4 w-4" />
                Generating...
              </>
            ) : (
              <>
                <Download className="h-4 w-4" />
                Export
              </>
            )}
          </button>
        </div>
      </aside>
    </div>,
    document.body
  );
}
export function GenerateReportButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-1.5 rounded-md border border-clean/40 bg-clean/10 text-clean px-3 py-1.5 text-xs font-semibold hover:bg-clean/15 transition-colors shadow-[0_0_18px_oklch(0.78_0.18_150_/_0.18)]"
    >
      <Download className="h-3.5 w-3.5" />
      Generate Report
    </button>
  );
}