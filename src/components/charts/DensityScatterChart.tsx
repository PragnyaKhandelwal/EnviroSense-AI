import { EChart } from "@/components/charts/EChart";
import { useSensorData } from "@/lib/resilience";

export function DensityScatterChart({ height = 220 }: { height?: number }) {
  const { snapshot } = useSensorData();

  const raw = Array.isArray(snapshot?.densityScatter)
    ? snapshot.densityScatter
    : [];

  const data = raw
    .map((d: any) => {
      // ✅ support BOTH formats
      const pm25 = d.pm2_5 ?? d.pm25;
      const pm10 = d.pm10_0 ?? d.pm10;

      if (typeof pm25 !== "number" || typeof pm10 !== "number") return null;

      return [pm25, pm10];
    })
    .filter(Boolean);

  if (!data.length) {
    return (
      <div className="text-xs text-muted-foreground">
        No scatter data available
      </div>
    );
  }

  return (
    <EChart
      option={{
        xAxis: { type: "value", name: "PM2.5" },
        yAxis: { type: "value", name: "PM10" },
        series: [
          {
            type: "scatter",
            data,
            symbolSize: 8,
            itemStyle: {
              opacity: 0.8,
            },
          },
        ],
      }}
      style={{ height }}
    />
  );
}