import { useSensorData } from "@/lib/resilience";
import {
  axisLine,
  baseGrid,
  baseTextStyle,
  baseTooltip,
  ENVIRO_COLORS,
  splitLine,
} from "@/lib/echarts-theme";
import { EChart } from "@/components/charts/EChart";

export function Timeline24hChart({ height = 320 }: { height?: number }) {
  const { snapshot } = useSensorData();

  if (!snapshot) {
    return <div style={{ padding: 20 }}>Loading chart...</div>;
  }

  const data = snapshot?.timeline24h || [];

  const option = {
    grid: { ...baseGrid, top: 36, right: 50 },

    xAxis: {
      type: "category",
      data: data.map((d: any) =>
        `${String(d?.hour ?? 0).padStart(2, "0")}:00`
      ),
      axisLine,
      axisLabel: { ...baseTextStyle, interval: 2 },
    },

    yAxis: [
      {
        type: "value",
        name: "µg/m³",
        axisLabel: baseTextStyle,
        splitLine,
      },
      {
        type: "value",
        name: "°C / %",
        axisLabel: baseTextStyle,
      },
    ],

    series: [
      {
        name: "PM2.5",
        type: "line",
        smooth: true,
        data: data.map((d: any) => d?.pm25 ?? 0),
      },
      {
        name: "Temperature",
        type: "line",
        yAxisIndex: 1,
        data: data.map((d: any) => d?.temperature ?? 0), // 🔥 FIXED
      },
      {
        name: "Humidity",
        type: "line",
        yAxisIndex: 1,
        data: data.map((d: any) => d?.humidity ?? 0),
      },
    ],
  };

  return <EChart option={option} style={{ height, width: "100%" }} />;
}
