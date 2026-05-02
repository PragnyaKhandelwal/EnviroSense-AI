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

export function ModelMetricsChart({ height = 220 }: { height?: number }) {
  const { snapshot, loading } = useSensorData();

  // ✅ guard render
  if (loading || !snapshot) {
    return <div style={{ padding: 10 }}>Loading...</div>;
  }

  const modelMetrics = snapshot?.modelMetrics || [];

  const option = {
    grid: { ...baseGrid, top: 32 },
    tooltip: { trigger: "axis", ...baseTooltip },
    legend: {
      data: ["MAE", "RMSE"],
      top: 0,
      textStyle: { color: ENVIRO_COLORS.axis, fontSize: 11 },
      icon: "roundRect",
    },
    xAxis: {
      type: "category",
      data: (modelMetrics || []).map((d: any) =>
        d?.t !== undefined ? d.t.toString() : ""
      ),
      axisLine,
      axisLabel: { ...baseTextStyle, interval: 3 },
    },
    yAxis: {
      type: "value",
      axisLine: { show: false },
      axisLabel: baseTextStyle,
      splitLine,
    },
    series: [
      {
        name: "MAE",
        type: "line",
        smooth: true,
        symbol: "none",
        data: (modelMetrics || []).map((d: any) => d?.mae ?? 0),
        lineStyle: { color: ENVIRO_COLORS.clean, width: 1.6 },
      },
      {
        name: "RMSE",
        type: "line",
        smooth: true,
        symbol: "none",
        data: (modelMetrics || []).map((d: any) => d?.rmse ?? 0),
        lineStyle: {
          color: ENVIRO_COLORS.cyan,
          width: 1.6,
          type: "dashed",
        },
      },
    ],
  };

  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
