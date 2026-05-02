import { AlertTriangle, CheckCircle2 } from "lucide-react";
import type { AnomalyState } from "@/lib/pipeline";

export function AnomalyStatusCard({ state }: { state: AnomalyState }) {
  const { anomalous, severity, detectedAt, code, title } = state;

  if (!anomalous) {
    return (
      <section className="panel panel-glow-clean p-5 h-full">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          Latest Anomaly Status
        </div>
        <div className="mt-3 flex items-center gap-3">
          <CheckCircle2 className="h-8 w-8 text-clean glow-text-clean" />
          <div>
            <div className="text-base font-semibold text-clean">All Clear</div>
            <div className="text-xs text-muted-foreground">Isolation Forest reports normal regime.</div>
          </div>
        </div>
        <div className="mt-3 text-[11px] text-muted-foreground">Last scan: 12s ago · 5,400 samples evaluated</div>
      </section>
    );
  }

  const tone = severity === "critical" ? "poor" : severity === "warning" ? "moderate" : "clean";
  const colorMap = {
    poor: { panel: "panel-glow-poor", text: "text-poor", glow: "glow-text-poor" },
    moderate: { panel: "panel-glow-moderate", text: "text-moderate", glow: "glow-text-moderate" },
    clean: { panel: "panel-glow-clean", text: "text-clean", glow: "glow-text-clean" },
  } as const;
  const c = colorMap[tone];

  return (
    <section className={`panel ${c.panel} p-5 h-full`}>
      <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Latest Anomaly Status</div>
      <div className="mt-3 flex items-start gap-3">
        <AlertTriangle className={`h-8 w-8 shrink-0 ${c.text} ${c.glow}`} />
        <div className="min-w-0">
          <div className={`text-base font-semibold ${c.text}`}>{title}</div>
          <div className="text-xs text-muted-foreground">
            Isolation Forest · class:{" "}
            <span className={`font-mono ${c.text}`}>{severity.toUpperCase()}</span> · 3.4σ deviation
          </div>
        </div>
      </div>
      <div className="mt-4 flex items-center justify-between text-[11px] text-muted-foreground border-t border-border pt-3">
        <span>Detected at <span className="font-mono text-foreground">{detectedAt}</span></span>
        <span>Code <span className="font-mono text-foreground">{code}</span></span>
      </div>
    </section>
  );
}
