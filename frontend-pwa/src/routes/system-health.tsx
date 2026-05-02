import { createFileRoute } from "@tanstack/react-router";
import { CheckCircle2, ShieldCheck, Activity, Cpu } from "lucide-react";
import { Layout } from "@/components/Layout";
import { PageHeader } from "@/components/PageHeader";
import { TrustGauge } from "@/components/charts/TrustGauge";
import { DriftControlChart } from "@/components/charts/DriftControlChart";
import { ModelMetricsChart } from "@/components/charts/ModelMetricsChart";
import { DeviceBoxplotChart } from "@/components/charts/DeviceBoxplotChart";
import { anomalies, modelRegistry } from "@/lib/mock-data";
import { aqiTextClass, type AqiLevel } from "@/lib/aqi";
import { SafeChart } from "@/components/SafeChart";
import { HistoricalBanner } from "@/components/HistoricalBanner";
import { useSensorData } from "@/lib/resilience";

export const Route = createFileRoute("/system-health")({
  head: () => ({
    meta: [
      { title: "Data Trust & System Health · EnviroSense AI" },
      {
        name: "description",
        content:
          "Quality metrics, drift detection control chart, model registry and anomaly timeline for the sensor infrastructure.",
      },
      { property: "og:title", content: "Data Trust & System Health · EnviroSense AI" },
      { property: "og:description", content: "Infrastructure layer telemetry." },
    ],
  }),
  component: SystemHealthPage,
});

function SystemHealthPage() {
  const { snapshot } = useSensorData();
  const { trust, uptime, validity } = snapshot.reliability;
  // Completeness derived from validity for now — same source signal.
  const completeness = +(validity + 0.8).toFixed(1);
  const validTone: "clean" | "moderate" | "poor" =
    validity >= 98 ? "clean" : validity >= 95 ? "moderate" : "poor";

  return (
    <Layout>
      <PageHeader
        eyebrow="Infrastructure Layer"
        title="Data Trust & System Health"
        description="Live quality telemetry, sensor drift detection and model registry performance."
      />

      <HistoricalBanner message="Trust telemetry frozen at last sync. Drift detector cannot evaluate new samples while offline." />

      {/* Quality grid */}
      <section className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <QualityCard label="Uptime" value={`${uptime.toFixed(1)}%`} icon={ShieldCheck} tone="clean" sub="Last 30 days" />
        <QualityCard label="Completeness" value={`${completeness.toFixed(1)}%`} icon={CheckCircle2} tone="clean" sub="Samples received / expected" />
        <QualityCard label="Valid %" value={`${validity.toFixed(1)}%`} icon={Activity} tone={validTone} sub="Passed validation rules" />
        <div className="panel p-4 flex flex-col">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Trust Score</div>
          <SafeChart label="trust-gauge" height={130}>
            <TrustGauge value={trust} height={130} label="Bayesian estimator" />
          </SafeChart>
        </div>
      </section>

      {/* Drift + model metrics */}
      <section className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="panel p-5 lg:col-span-2">
          <div className="flex items-end justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                Drift Detection
              </div>
              <div className="mt-0.5 text-sm">Z-score control chart · ±2σ control limits</div>
            </div>
            <span className="inline-flex items-center gap-1.5 rounded-md border border-moderate/40 bg-moderate/10 text-moderate px-2 py-1 text-[10px] font-medium uppercase tracking-wider">
              Drifting
            </span>
          </div>
          <div className="mt-3">
            <SafeChart label="drift" height={280}>
              <DriftControlChart />
            </SafeChart>
          </div>
        </div>

        <div className="panel p-5">
          <div className="flex items-end justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                Model Performance
              </div>
              <div className="mt-0.5 text-sm">Rolling MAE / RMSE</div>
            </div>
            <Cpu className="h-4 w-4 text-cyan" />
          </div>
          <SafeChart label="model-metrics" height={220}>
            <ModelMetricsChart />
          </SafeChart>
        </div>
      </section>

      {/* Model registry + device comparison */}
      <section className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="panel p-5 lg:col-span-2">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Model Registry</div>
          <div className="mt-0.5 text-sm">Active models powering the intelligence layer</div>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-[10px] uppercase tracking-wider text-muted-foreground">
                <tr className="border-b border-border">
                  <th className="text-left font-medium py-2">Model</th>
                  <th className="text-left font-medium py-2">Version</th>
                  <th className="text-right font-medium py-2">MAE</th>
                  <th className="text-right font-medium py-2">RMSE</th>
                  <th className="text-right font-medium py-2">Status</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {modelRegistry.map((m) => (
                  <tr key={m.name} className="border-b border-border/50">
                    <td className="py-2.5">{m.name}</td>
                    <td className="py-2.5 text-muted-foreground">{m.version}</td>
                    <td className="py-2.5 text-right">{m.mae.toFixed(2)}</td>
                    <td className="py-2.5 text-right">{m.rmse.toFixed(2)}</td>
                    <td className="py-2.5 text-right">
                      <span
                        className={`inline-block rounded-full px-2 py-0.5 text-[10px] uppercase ${
                          m.status === "active"
                            ? "bg-clean/10 text-clean border border-clean/30"
                            : "bg-moderate/10 text-moderate border border-moderate/30"
                        }`}
                      >
                        {m.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="panel p-5">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Device Comparison</div>
          <div className="mt-0.5 text-sm">Sensor 1 vs Sensor 2 · boxplot</div>
          <SafeChart label="device-boxplot" height={320}>
            <DeviceBoxplotChart />
          </SafeChart>
        </div>
      </section>

      {/* Anomalies timeline */}
      <section className="mt-4 panel p-5">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Anomalies Timeline</div>
        <div className="mt-0.5 text-sm">Log-style stream · most recent first</div>
        <ul className="mt-4 divide-y divide-border font-mono text-xs">
          {anomalies.map((a) => (
            <li key={a.ts + a.code} className="grid grid-cols-[140px_1fr_auto] items-center gap-3 py-2.5">
              <span className="text-muted-foreground">{a.ts}</span>
              <span className={aqiTextClass(a.level as AqiLevel)}>{a.label}</span>
              <span className="rounded-md border border-border px-2 py-0.5 text-[10px] text-muted-foreground">
                {a.code}
              </span>
            </li>
          ))}
        </ul>
      </section>
    </Layout>
  );
}

function QualityCard({
  label,
  value,
  icon: Icon,
  tone,
  sub,
}: {
  label: string;
  value: string;
  icon: React.ComponentType<{ className?: string }>;
  tone: "clean" | "moderate" | "poor";
  sub?: string;
}) {
  const colorMap = {
    clean: "text-clean",
    moderate: "text-moderate",
    poor: "text-poor",
  } as const;
  return (
    <div className="panel p-4">
      <div className="flex items-center justify-between">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
        <Icon className={`h-4 w-4 ${colorMap[tone]}`} />
      </div>
      <div className={`mt-2 font-mono text-3xl font-semibold ${colorMap[tone]}`}>{value}</div>
      {sub && <div className="mt-1 text-[10px] text-muted-foreground">{sub}</div>}
    </div>
  );
}
