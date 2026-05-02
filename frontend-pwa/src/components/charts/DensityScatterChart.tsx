import { EChart } from "@/components/charts/EChart";
import {
  axisLine,
  baseGrid,
  baseTextStyle,
  baseTooltip,
  ENVIRO_COLORS,
  splitLine,
} from "@/lib/echarts-theme";
import { densityScatter } from "@/lib/mock-data";

const COLORS = [
  ENVIRO_COLORS.clean,
  ENVIRO_COLORS.cyan,
  ENVIRO_COLORS.moderate,
  ENVIRO_COLORS.amber,
  ENVIRO_COLORS.poor,
];

export function DensityScatterChart({
  binIndex,
  height = 220,
}: {
  binIndex: number;
  height?: number;
}) {
  const bin = densityScatter[binIndex];
  const color = COLORS[binIndex % COLORS.length];

  const option = {
    grid: { ...baseGrid, top: 18, left: 36, bottom: 32 },
    tooltip: {
      ...baseTooltip,
      trigger: "item",
      formatter: (p: { value: [number, number] }) =>
        `Count: <b>${p.value[0]}</b><br/>Mass: <b>${p.value[1]}</b>`,
    },
    xAxis: {
      type: "value",
      name: "Count",
      nameTextStyle: baseTextStyle,
      nameLocation: "middle",
      nameGap: 22,
      axisLine,
      axisLabel: baseTextStyle,
      splitLine,
    },
    yAxis: {
      type: "value",
      name: "Mass",
      nameTextStyle: baseTextStyle,
      axisLine: { show: false },
      axisLabel: baseTextStyle,
      splitLine,
    },
    series: [
      {
        type: "scatter",
        symbolSize: 6,
        data: bin.points.map((p) => [p.count, p.mass]),
        itemStyle: { color, opacity: 0.75 },
      },
    ],
  };

  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
