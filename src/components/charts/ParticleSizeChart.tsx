import { EChart } from "@/components/charts/EChart";
import { useSensorData } from "@/lib/resilience";

type ParticleRow = {
  pm25: number;
  pm10: number;
};

type ParticlePoint = {
  t: number;
  "0-0.3µm": number;
  "0.3-1µm": number;
  "1-2.5µm": number;
  "2.5-5µm": number;
  "5-10µm": number;
};

export function ParticleSizeChart({ height = 320 }: { height?: number }) {
  const { snapshot } = useSensorData();

  const raw: ParticleRow[] = Array.isArray(snapshot?.particleSize)
    ? snapshot.particleSize
    : [];

  const data: ParticlePoint[] = raw
    .filter(
      (d) =>
        d &&
        typeof d.pm25 === "number" &&
        typeof d.pm10 === "number"
    )
    .map((d, i): ParticlePoint => ({
      t: i,
      "0-0.3µm": d.pm25 * 0.2,
      "0.3-1µm": d.pm25 * 0.3,
      "1-2.5µm": d.pm25 * 0.25,
      "2.5-5µm": d.pm25 * 0.15,
      "5-10µm": d.pm10 * 0.1,
    }));

  if (!data.length) {
    return (
      <div className="text-xs text-muted-foreground">
        No particle data available
      </div>
    );
  }

  const bins = [
    "0-0.3µm",
    "0.3-1µm",
    "1-2.5µm",
    "2.5-5µm",
    "5-10µm",
  ];

  return (
    <EChart
      option={{
        xAxis: {
          type: "category",
          data: data.map((d) => d.t),
        },
        yAxis: { type: "value" },
        series: bins.map((b) => ({
          name: b,
          type: "line",
          stack: "size",
          areaStyle: {},
          data: data.map((d) => d[b as keyof ParticlePoint] ?? 0),
        })),
      }}
      style={{ height }}
    />
  );
}