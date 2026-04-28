import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { TrendingUp, ArrowUpRight, Activity } from "lucide-react";
import { Layout } from "@/components/Layout";
import { PageHeader } from "@/components/PageHeader";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { RegimeTransitionMatrix } from "@/components/charts/RegimeTransitionMatrix";
import { ParticleSizeChart } from "@/components/charts/ParticleSizeChart";
import { DensityScatterChart } from "@/components/charts/DensityScatterChart";
import { CorrelationScatter } from "@/components/charts/CorrelationScatter";
import { DiurnalHeatmapChart } from "@/components/charts/DiurnalHeatmapChart";
import { sizeBins, sizeScenarios, densityScatter } from "@/lib/mock-data";
import { SafeChart } from "@/components/SafeChart";
import { HistoricalBanner } from "@/components/HistoricalBanner";
import { GenerateReportButton, ReportDrawer } from "@/components/ReportDrawer";
import { useSensorData } from "@/lib/resilience";

export const Route = createFileRoute("/predictive")({
  head: () => ({
    meta: [
      { title: "Predictive Intelligence · EnviroSense AI" },
      {
        name: "description",
        content:
          "Short-term PM2.5 forecasts with confidence bands, temporal momentum, regime transition matrix, particle physics and correlations.",
      },
      { property: "og:title", content: "Predictive Intelligence · EnviroSense AI" },
      { property: "og:description", content: "Forecasts, regimes & data physics." },
    ],
  }),
  component: PredictivePage,
});

const SCENARIOS = ["Normal", "Dust Storm", "Rain", "Post-Rain Clean Air"] as const;
const TABS = ["Forecast", "Particle Physics", "Correlations", "Diurnal"] as const;

function PredictivePage() {
  const { snapshot } = useSensorData();
  const [tab, setTab] = useState<(typeof TABS)[number]>("Forecast");
  const [scenario, setScenario] = useState<keyof typeof sizeScenarios>("Normal");
  const [axis, setAxis] = useState<"humidity" | "temp">("humidity");
  const [metric, setMetric] = useState<"pm25" | "pm10">("pm25");
  const [reportOpen, setReportOpen] = useState(false);
  const currentRegime = snapshot.regime.current;

  return (
    <Layout>
      <PageHeader
        eyebrow="ML Layer"
        title="Predictive Intelligence"
        description="Short-term forecasts, regime transitions, and the physics of particles driving today's measurements."
        actions={<GenerateReportButton onClick={() => setReportOpen(true)} />}
      />
      <ReportDrawer open={reportOpen} onClose={() => setReportOpen(false)} />

      {/* Tabs */}
      <div className="inline-flex flex-wrap rounded-lg border border-border bg-panel p-0.5 text-xs mb-4">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-1.5 rounded-md transition-colors ${
              tab === t ? "bg-clean/15 text-clean" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      <HistoricalBanner message="Viewing cached AI forecast. Live inference paused." />

      {tab === "Forecast" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <section className="panel p-5 lg:col-span-2">
            <div className="flex flex-wrap items-end justify-between gap-2">
              <div>
                <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                  Short-Term Forecast
                </div>
                <div className="mt-0.5 text-sm">PM2.5 · history (solid) → next 60 min (dashed) with 95% CI</div>
              </div>
              <span className="inline-flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-[10px] uppercase tracking-widest text-muted-foreground">
                Model: XGBoost v2.1
              </span>
            </div>
            <div className="mt-3">
              <SafeChart label="forecast" height={360}>
                <ForecastChart />
              </SafeChart>
            </div>
          </section>

          <section className="panel p-5 flex flex-col gap-4">
            <div>
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                Temporal Momentum
              </div>
              <div className="mt-0.5 text-sm">Trend strength · pattern classification</div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-clean/30 bg-clean/5 p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Slope</div>
                <div className="mt-1 flex items-baseline gap-1">
                  <TrendingUp className="h-4 w-4 text-clean" />
                  <span className="font-mono text-2xl font-semibold text-clean glow-text-clean">+4.2</span>
                </div>
                <div className="mt-1 text-[10px] text-muted-foreground">µg/m³ per 5 min</div>
              </div>
              <div className="rounded-lg border border-border p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Half-life</div>
                <div className="mt-1 flex items-baseline gap-1">
                  <Activity className="h-4 w-4 text-cyan" />
                  <span className="font-mono text-2xl font-semibold">18m</span>
                </div>
                <div className="mt-1 text-[10px] text-muted-foreground">Decay estimate</div>
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Pattern</div>
              <div className="mt-1 inline-flex items-center gap-1.5 rounded-full border border-poor/40 bg-poor/10 text-poor px-3 py-1 text-xs font-medium">
                <ArrowUpRight className="h-3.5 w-3.5" /> Spike
              </div>
              <p className="mt-2 text-[11px] text-muted-foreground leading-relaxed">
                Rapid acceleration with positive slope. Likely transient — expect recovery within 20–30 minutes.
              </p>
            </div>
          </section>

          <section className="panel p-5 lg:col-span-3">
            <div className="flex items-end justify-between">
              <div>
                <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                  Regime Transition Matrix
                </div>
                <div className="mt-0.5 text-sm">
                  HMM · probability of moving from <span className="text-clean">{currentRegime}</span> to next state
                </div>
              </div>
              <span className="text-[10px] text-muted-foreground hidden sm:inline">Highlighted row = current state</span>
            </div>
            <div className="mt-3">
              <SafeChart label="regime-matrix" height={320}>
                <RegimeTransitionMatrix currentRegime={currentRegime} />
              </SafeChart>
            </div>
          </section>
        </div>
      )}

      {tab === "Particle Physics" && (
        <div className="space-y-4">
          <section className="panel p-5">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                  Particle Size Distribution
                </div>
                <div className="mt-0.5 text-sm">Stacked bin profile · {scenario}</div>
              </div>
              <div className="inline-flex flex-wrap rounded-lg border border-border bg-panel p-0.5 text-xs">
                {SCENARIOS.map((s) => (
                  <button
                    key={s}
                    onClick={() => setScenario(s)}
                    className={`px-3 py-1.5 rounded-md transition-colors ${
                      scenario === s ? "bg-clean/15 text-clean" : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-3">
              <SafeChart label="particle-size" height={320} resetKey={scenario}>
                <ParticleSizeChart scenario={scenario} />
              </SafeChart>
            </div>
          </section>

          <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {densityScatter.slice(0, 3).map((s, i) => (
              <div key={s.bin} className="panel p-5">
                <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Density Factor</div>
                <div className="mt-0.5 text-sm">Count vs Mass · bin {s.bin}</div>
                <div className="mt-3">
                  <SafeChart label={`density-${i}`} height={220}>
                    <DensityScatterChart binIndex={i} />
                  </SafeChart>
                </div>
              </div>
            ))}
          </section>

          <div className="text-[11px] text-muted-foreground flex flex-wrap gap-3">
            {sizeBins.map((b) => (
              <span key={b}>{b}</span>
            ))}
          </div>
        </div>
      )}

      {tab === "Correlations" && (
        <section className="panel p-5">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                Environmental Correlations
              </div>
              <div className="mt-0.5 text-sm">
                PM2.5 vs {axis === "humidity" ? "Humidity" : "Temperature"} · color-coded by AQI cluster
              </div>
            </div>
            <div className="inline-flex rounded-lg border border-border bg-panel p-0.5 text-xs">
              {(["humidity", "temp"] as const).map((a) => (
                <button
                  key={a}
                  onClick={() => setAxis(a)}
                  className={`px-3 py-1.5 rounded-md transition-colors ${
                    axis === a ? "bg-clean/15 text-clean" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {a === "humidity" ? "Humidity" : "Temperature"}
                </button>
              ))}
            </div>
          </div>
          <div className="mt-3">
            <SafeChart label="correlation" height={420} resetKey={axis}>
              <CorrelationScatter axis={axis} />
            </SafeChart>
          </div>
        </section>
      )}

      {tab === "Diurnal" && (
        <section className="panel p-5">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                Weekly Diurnal Heatmap
              </div>
              <div className="mt-0.5 text-sm">Hour × Day · color intensity = pollutant level</div>
            </div>
            <div className="inline-flex rounded-lg border border-border bg-panel p-0.5 text-xs">
              {(["pm25", "pm10"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMetric(m)}
                  className={`px-3 py-1.5 rounded-md transition-colors ${
                    metric === m ? "bg-clean/15 text-clean" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {m === "pm25" ? "PM2.5" : "PM10"}
                </button>
              ))}
            </div>
          </div>
          <div className="mt-3">
            <SafeChart label="diurnal" height={320} resetKey={metric}>
              <DiurnalHeatmapChart metric={metric} />
            </SafeChart>
          </div>
        </section>
      )}
    </Layout>
  );
}
