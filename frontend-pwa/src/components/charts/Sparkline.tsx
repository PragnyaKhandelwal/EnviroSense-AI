import { EChart } from "@/components/charts/EChart";
import { ENVIRO_COLORS } from "@/lib/echarts-theme";

export function Sparkline({
  data,
  color = ENVIRO_COLORS.clean,
  height = 44,
}: {
  data: number[];
  color?: string;
  height?: number;
}) {
  const safe = Array.isArray(data) ? data.filter((v) => typeof v === "number") : [];
  const option = {
    grid: { left: 0, right: 0, top: 4, bottom: 0 },
    xAxis: { type: "category", show: false, boundaryGap: false, data: safe.map((_, i) => i) },
    yAxis: { type: "value", show: false, scale: true },
    tooltip: {
      trigger: "axis",
      backgroundColor: ENVIRO_COLORS.bg,
      borderColor: ENVIRO_COLORS.border,
      textStyle: { color: ENVIRO_COLORS.text, fontSize: 11 },
      formatter: (params: { value: number }[]) =>
        `${params?.[0]?.value?.toFixed?.(1) ?? "—"} µg/m³`,
    },
    series: [
      {
        type: "line",
        smooth: true,
        data: safe,
        symbol: "none",
        lineStyle: { color, width: 1.5 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: color.replace(")", " / 0.45)").replace("oklch(", "oklch(") },
              { offset: 1, color: color.replace(")", " / 0)").replace("oklch(", "oklch(") },
            ],
          },
        },
      },
    ],
  };
  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
