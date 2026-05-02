import { TrendingUp, Snowflake } from "lucide-react";
import { classifyPM, aqiLabel, aqiPanelClass, aqiTextClass, aqiGlowText } from "@/lib/aqi";
import { Sparkline } from "@/components/charts/Sparkline";
import { ENVIRO_COLORS } from "@/lib/echarts-theme";
import { useSensorData, formatRelative } from "@/lib/resilience";
import { SafeChart } from "@/components/SafeChart";

/** Hero "Live Status" card with PM2.5 / PM10, sparkline, and city comparison */
export function LiveStatusCard() {
  const { snapshot, isLive, lastSyncTs } = useSensorData();
  const { pm25, pm10, cityAvg, delta, sparkline, device_label } = snapshot.sensor;

  const level25 = classifyPM(pm25);
  const level10 = classifyPM(pm10);
  const cityLevel = classifyPM(cityAvg);
  const sparkColor =
    level25 === "clean" ? ENVIRO_COLORS.clean : level25 === "moderate" ? ENVIRO_COLORS.moderate : ENVIRO_COLORS.poor;

  // Safe array — never let a malformed payload crash the sparkline
  const safeSpark = Array.isArray(sparkline) && sparkline.length > 0 ? sparkline : [];

  return (
    <section className={`relative panel ${aqiPanelClass(level25)} p-5 sm:p-6 overflow-hidden ${isLive ? "scanline" : ""} h-full`}>
      <div className="grid-bg absolute inset-0 opacity-[0.08] pointer-events-none" />
      <div className="relative flex items-start justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Live Status</div>
          <div className="mt-0.5 text-sm font-medium">{device_label}</div>
        </div>
        <div className="text-right">
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
            {isLive ? "Micro-batch" : "Cached"}
          </div>
          <div className="font-mono text-xs flex items-center gap-1 justify-end">
            {isLive ? (
              <>
                <span className={aqiTextClass(level25)}>●</span> {formatRelative(lastSyncTs)}
              </>
            ) : (
              <>
                <Snowflake className="h-3 w-3 text-moderate" />
                <span className="text-moderate">{formatRelative(lastSyncTs)}</span>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="relative mt-5 grid grid-cols-2 gap-6">
        <Reading label="PM2.5" unit="µg/m³" value={pm25} level={level25} delta={delta} />
        <Reading label="PM10" unit="µg/m³" value={pm10} level={level10} />
      </div>

      <div className="relative mt-4">
        <SafeChart label="hero-sparkline" height={48}>
          <Sparkline data={safeSpark} color={sparkColor} height={48} />
        </SafeChart>
        <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
          <span>−60 min</span>
          <span>{isLive ? "now" : "last sync"}</span>
        </div>
      </div>

      <div className="relative mt-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-t border-border pt-3">
        <div className="text-sm">
          <span className="text-muted-foreground">Cluster:</span>{" "}
          <span className={`font-medium ${aqiTextClass(level25)}`}>{aqiLabel(level25)}</span>
        </div>
        <div className="text-xs text-muted-foreground">
          City Avg:{" "}
          <span className={`font-medium ${aqiTextClass(cityLevel)}`}>
            {cityAvg} ({cityLevel === "clean" ? "Green" : cityLevel === "moderate" ? "Yellow" : "Red"})
          </span>
        </div>
      </div>
    </section>
  );
}

function Reading({
  label,
  unit,
  value,
  level,
  delta,
}: {
  label: string;
  unit: string;
  value: number;
  level: ReturnType<typeof classifyPM>;
  delta?: number;
}) {
  return (
    <div>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        {label}
        {typeof delta === "number" && (
          <span className={`inline-flex items-center gap-0.5 rounded px-1 py-0.5 text-[10px] ${delta >= 0 ? "text-poor" : "text-clean"}`}>
            <TrendingUp className={`h-3 w-3 ${delta < 0 ? "rotate-180" : ""}`} />
            {Math.abs(delta).toFixed(1)}
          </span>
        )}
      </div>
      <div
        className={`mt-1 font-mono text-4xl sm:text-5xl font-semibold tabular-nums ${aqiTextClass(level)} ${aqiGlowText(level)}`}
      >
        {value.toFixed(1)}
      </div>
      <div className="mt-0.5 text-[11px] text-muted-foreground">{unit}</div>
    </div>
  );
}
