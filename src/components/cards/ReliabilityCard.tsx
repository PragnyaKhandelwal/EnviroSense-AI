import { TrustGauge } from "@/components/charts/TrustGauge";
import { ShieldCheck } from "lucide-react";
import { useSensorData } from "@/lib/resilience";

// ✅ Move helper OUTSIDE component
function Stat({ label, v }: { label: string; v: string }) {
  return (
    <div className="text-center">
      <div className="font-mono text-xs text-foreground">{v}</div>
      <div>{label}</div>
    </div>
  );
}

export function ReliabilityCard() {
  const { snapshot, loading } = useSensorData();

  // ✅ safe loading check INSIDE component
  if (loading || !snapshot) {
    return <div className="p-4">Loading...</div>;
  }

  const reliability = snapshot?.reliability ?? {};

  const trust = reliability?.trust ?? 0;
  const uptime = reliability?.uptime ?? 0;
  const validity = reliability?.validity ?? 0;
  const driftSigma = reliability?.driftSigma ?? 0;

  // ✅ fixed ternary
  const trustLabel =
    trust >= 90
      ? "High Confidence"
      : trust >= 75
      ? "Moderate Confidence"
      : "Low Confidence";

  const safe = (v: number | undefined, unit: string = "") =>
    typeof v === "number" ? `${v.toFixed(1)}${unit}` : "—";

  return (
    <section className="panel p-5 h-full flex flex-col">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Reliability Score
          </div>
          <div className="mt-0.5 text-sm">
            Trust Layer · Bayesian estimator
          </div>
        </div>
        <ShieldCheck className="h-4 w-4 text-clean" />
      </div>

      <div className="flex-1">
        <TrustGauge value={trust} height={180} label={trustLabel} />
      </div>

      <div className="mt-2 grid grid-cols-3 gap-2 text-[10px] text-muted-foreground border-t border-border pt-3">
        <Stat label="Uptime" v={safe(uptime, "%")} />
        <Stat label="Valid" v={safe(validity, "%")} />
        <Stat label="Drift σ" v={safe(driftSigma)} />
      </div>
    </section>
  );
}
