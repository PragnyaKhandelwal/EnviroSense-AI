import { EChart } from "@/components/charts/EChart";
import {
  ENVIRO_COLORS,
  axisLine,
  baseGrid,
  baseTextStyle,
  baseTooltip,
  splitLine,
} from "@/lib/echarts-theme";
import { useSensorData } from "@/lib/resilience";

/** 24-hour timeline: PM2.5 (left axis) overlaid with Temp & Humidity (right axis) */
export function Timeline24hChart({ height = 320 }: { height?: number }) {
  const { snapshot } = useSensorData();
  const data = snapshot.timeline24h;

  const option = {
    grid: { ...baseGrid, top: 36, right: 50 },
    legend: {
      data: ["PM2.5", "Temperature", "Humidity"],
      textStyle: { color: ENVIRO_COLORS.axis, fontSize: 11 },
      top: 0,
      right: 8,
      icon: "roundRect",
    },
    tooltip: { trigger: "axis", ...baseTooltip },
    xAxis: {
      type: "category",
      data: data.map((d) => `${String(d.hour).padStart(2, "0")}:00`),
      axisLine,
      axisLabel: { ...baseTextStyle, interval: 2 },
    },
    yAxis: [
      {
        type: "value",
        name: "µg/m³",
        position: "left",
        nameTextStyle: { ...baseTextStyle },
        axisLine: { show: false },
        axisLabel: baseTextStyle,
        splitLine,
      },
      {
        type: "value",
        name: "°C / %",
        position: "right",
        nameTextStyle: { ...baseTextStyle },
        axisLine: { show: false },
        axisLabel: baseTextStyle,
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: "PM2.5",
        type: "line",
        smooth: true,
        symbol: "none",
        data: data.map((d) => d.pm25),
        lineStyle: { color: ENVIRO_COLORS.clean, width: 2.2 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: "oklch(0.78 0.18 150 / 0.45)" },
              { offset: 1, color: "oklch(0.78 0.18 150 / 0)" },
            ],
          },
        },
      },
      {
        name: "Temperature",
        type: "line",
        smooth: true,
        symbol: "none",
        yAxisIndex: 1,
        data: data.map((d) => d.temp),
        lineStyle: { color: ENVIRO_COLORS.amber, width: 1.6, type: "dashed" },
      },
      {
        name: "Humidity",
        type: "line",
        smooth: true,
        symbol: "none",
        yAxisIndex: 1,
        data: data.map((d) => d.humidity),
        lineStyle: { color: ENVIRO_COLORS.violet, width: 1.6, type: "dashed" },
      },
    ],
  };
  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
