import { EChart } from "@/components/charts/EChart";
import {
  axisLine,
  baseGrid,
  baseTextStyle,
  baseTooltip,
  ENVIRO_COLORS,
  splitLine,
} from "@/lib/echarts-theme";
import { heatmapPM10, heatmapPM25 } from "@/lib/mock-data";

export function DiurnalHeatmapChart({
  metric,
  height = 320,
}: {
  metric: "pm25" | "pm10";
  height?: number;
}) {
  const data = metric === "pm25" ? heatmapPM25 : heatmapPM10;
  const series: Array<[number, number, number]> = [];
  data.forEach((row, di) => {
    row.hours.forEach((h) => {
      series.push([h.hour, di, h.value]);
    });
  });
  const max = Math.max(...series.map((s) => s[2]));

  const option = {
    grid: { ...baseGrid, top: 16, bottom: 70, left: 36 },
    tooltip: {
      ...baseTooltip,
      formatter: (p: { value: [number, number, number] }) =>
        `${data[p.value[1]].day} ${String(p.value[0]).padStart(2, "0")}:00<br/><b>${p.value[2]}</b> µg/m³`,
    },
    xAxis: {
      type: "category",
      data: Array.from({ length: 24 }, (_, h) => h.toString()),
      axisLine,
      axisLabel: { ...baseTextStyle, interval: 2 },
      splitArea: { show: false },
    },
    yAxis: {
      type: "category",
      data: data.map((d) => d.day),
      inverse: true,
      axisLine: { show: false },
      axisLabel: baseTextStyle,
      splitArea: { show: false },
      splitLine,
    },
    visualMap: {
      min: 0,
      max,
      calculable: false,
      orient: "horizontal",
      left: "center",
      bottom: 8,
      itemWidth: 10,
      itemHeight: 120,
      textStyle: { color: ENVIRO_COLORS.axis, fontSize: 10 },
      inRange: {
        color: [
          "oklch(0.78 0.18 150 / 0.25)",
          ENVIRO_COLORS.clean,
          ENVIRO_COLORS.moderate,
          ENVIRO_COLORS.poor,
        ],
      },
    },
    series: [
      {
        type: "heatmap",
        data: series,
        itemStyle: { borderColor: ENVIRO_COLORS.bg, borderWidth: 1 },
        emphasis: { itemStyle: { borderColor: ENVIRO_COLORS.clean, borderWidth: 1.5 } },
      },
    ],
  };

  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
