import { EChart } from "@/components/charts/EChart";
import {
  axisLine,
  baseGrid,
  baseTextStyle,
  baseTooltip,
  ENVIRO_COLORS,
  splitLine,
} from "@/lib/echarts-theme";
import { sizeBins, sizeScenarios } from "@/lib/mock-data";

const BIN_COLORS = [
  ENVIRO_COLORS.clean,
  ENVIRO_COLORS.cyan,
  ENVIRO_COLORS.moderate,
  ENVIRO_COLORS.amber,
  ENVIRO_COLORS.poor,
];

export function ParticleSizeChart({
  scenario,
  height = 320,
}: {
  scenario: keyof typeof sizeScenarios;
  height?: number;
}) {
  const data = sizeScenarios[scenario];

  const option = {
    grid: { ...baseGrid, top: 32 },
    tooltip: { trigger: "axis", ...baseTooltip, axisPointer: { type: "shadow" } },
    legend: {
      data: sizeBins,
      textStyle: { color: ENVIRO_COLORS.axis, fontSize: 10 },
      top: 0,
      icon: "roundRect",
    },
    xAxis: {
      type: "category",
      data: data.map((d) => d.t.toString()),
      axisLine,
      axisLabel: { ...baseTextStyle, interval: 3 },
    },
    yAxis: {
      type: "value",
      axisLine: { show: false },
      axisLabel: baseTextStyle,
      splitLine,
    },
    series: sizeBins.map((b, i) => ({
      name: b,
      type: "line",
      stack: "size",
      smooth: true,
      symbol: "none",
      data: data.map((d) => d[b] as number),
      lineStyle: { color: BIN_COLORS[i], width: 1 },
      areaStyle: { color: BIN_COLORS[i], opacity: 0.55 },
    })),
  };

  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
