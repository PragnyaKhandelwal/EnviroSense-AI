import { EChart } from "@/components/charts/EChart";
import {
  axisLine,
  baseGrid,
  baseTextStyle,
  baseTooltip,
  ENVIRO_COLORS,
  splitLine,
} from "@/lib/echarts-theme";
import { useSensorData } from "@/lib/resilience";

/** Z-score control chart with ±2σ control limits */
export function DriftControlChart({ height = 280 }: { height?: number }) {
  const { snapshot } = useSensorData();
  const driftSeries = snapshot.drift;
  const option = {
    grid: { ...baseGrid, top: 32 },
    tooltip: {
      trigger: "axis",
      ...baseTooltip,
      formatter: (p: Array<{ axisValue: string; value: number }>) =>
        `t=${p[0].axisValue} · z=${p[0].value.toFixed(2)}`,
    },
    xAxis: {
      type: "category",
      data: driftSeries.map((_, i) => i.toString()),
      axisLine,
      axisLabel: { ...baseTextStyle, interval: 9 },
    },
    yAxis: {
      type: "value",
      name: "Z-score",
      nameTextStyle: baseTextStyle,
      axisLine: { show: false },
      axisLabel: baseTextStyle,
      splitLine,
      min: -3,
      max: 3,
    },
    series: [
      {
        type: "line",
        data: driftSeries,
        smooth: true,
        symbol: "circle",
        symbolSize: 4,
        lineStyle: { color: ENVIRO_COLORS.cyan, width: 1.6 },
        itemStyle: { color: ENVIRO_COLORS.cyan },
        markLine: {
          symbol: "none",
          silent: true,
          data: [
            { yAxis: 2, lineStyle: { color: ENVIRO_COLORS.poor, type: "dashed" }, label: { formatter: "+2σ", color: ENVIRO_COLORS.poor, fontSize: 10 } },
            { yAxis: -2, lineStyle: { color: ENVIRO_COLORS.poor, type: "dashed" }, label: { formatter: "−2σ", color: ENVIRO_COLORS.poor, fontSize: 10 } },
            { yAxis: 0, lineStyle: { color: ENVIRO_COLORS.border }, label: { show: false } },
          ],
        },
        markArea: {
          silent: true,
          itemStyle: { color: "oklch(0.68 0.24 25 / 0.06)" },
          data: [[{ yAxis: 2 }, { yAxis: 3 }], [{ yAxis: -3 }, { yAxis: -2 }]],
        },
      },
    ],
  };
  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
