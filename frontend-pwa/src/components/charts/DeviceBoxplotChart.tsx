import { EChart } from "@/components/charts/EChart";
import {
  axisLine,
  baseGrid,
  baseTextStyle,
  baseTooltip,
  ENVIRO_COLORS,
  splitLine,
} from "@/lib/echarts-theme";
import { deviceStats } from "@/lib/mock-data";

/** Native ECharts boxplot comparing Sensor 1 vs Sensor 2 across 6 hourly windows */
export function DeviceBoxplotChart({ height = 320 }: { height?: number }) {
  const categories = deviceStats.map((d) => `${d.hour}h`);
  const s1 = deviceStats.map((d) => [d.s1.min, d.s1.q1, d.s1.med, d.s1.q3, d.s1.max]);
  const s2 = deviceStats.map((d) => [d.s2.min, d.s2.q1, d.s2.med, d.s2.q3, d.s2.max]);

  const option = {
    grid: { ...baseGrid, top: 32 },
    legend: {
      top: 0,
      data: ["Sensor 1", "Sensor 2"],
      textStyle: { color: ENVIRO_COLORS.axis, fontSize: 11 },
      icon: "roundRect",
    },
    tooltip: { ...baseTooltip, trigger: "item" },
    xAxis: {
      type: "category",
      data: categories,
      axisLine,
      axisLabel: baseTextStyle,
    },
    yAxis: {
      type: "value",
      name: "µg/m³",
      nameTextStyle: baseTextStyle,
      axisLine: { show: false },
      axisLabel: baseTextStyle,
      splitLine,
    },
    series: [
      {
        name: "Sensor 1",
        type: "boxplot",
        data: s1,
        itemStyle: { color: "oklch(0.78 0.18 150 / 0.25)", borderColor: ENVIRO_COLORS.clean },
      },
      {
        name: "Sensor 2",
        type: "boxplot",
        data: s2,
        itemStyle: { color: "oklch(0.80 0.14 200 / 0.25)", borderColor: ENVIRO_COLORS.cyan },
      },
    ],
  };

  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
