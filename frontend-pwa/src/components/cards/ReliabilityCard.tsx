import { TrustGauge } from "@/components/charts/TrustGauge";
import { ShieldCheck } from "lucide-react";
import { useSensorData } from "@/lib/resilience";

export function ReliabilityCard() {
  const { snapshot } = useSensorData();
  const { trust, uptime, validity, driftSigma } = snapshot.reliability;

  const trustLabel =
    trust >= 90 ? "High Confidence" : trust >= 75 ? "Moderate Confidence" : "Low Confidence";

  return (
    <section className="panel p-5 h-full flex flex-col">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Reliability Score</div>
          <div className="mt-0.5 text-sm">Trust Layer · Bayesian estimator</div>
        </div>
        <ShieldCheck className="h-4 w-4 text-clean" />
      </div>
      <div className="flex-1">
        <TrustGauge value={trust} height={180} label={trustLabel} />
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2 text-[10px] text-muted-foreground border-t border-border pt-3">
        <Stat label="Uptime" v={`${uptime.toFixed(1)}%`} />
        <Stat label="Valid" v={`${validity.toFixed(1)}%`} />
        <Stat label="Drift" v={`${driftSigma.toFixed(1)}σ`} />
      </div>
    </section>
  );
}

function Stat({ label, v }: { label: string; v: string }) {
  return (
    <div className="text-center">
      <div className="font-mono text-xs text-foreground">{v}</div>
      <div>{label}</div>
    </div>
  );
}
