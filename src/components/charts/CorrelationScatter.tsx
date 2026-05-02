import { EChart } from "@/components/charts/EChart";
import {
  axisLine,
  baseGrid,
  baseTextStyle,
  baseTooltip,
  ENVIRO_COLORS,
  splitLine,
} from "@/lib/echarts-theme";
import { correlationData } from "@/lib/mock-data";
import { classifyPM } from "@/lib/aqi";

export function CorrelationScatter({
  axis,
  height = 420,
}: {
  axis: "humidity" | "temp";
  height?: number;
}) {
  const xLabel = axis === "humidity" ? "Humidity (%)" : "Temperature (°C)";

  const buckets = {
    clean: [] as Array<[number, number]>,
    moderate: [] as Array<[number, number]>,
    poor: [] as Array<[number, number]>,
  };
  correlationData.forEach((d) => {
    const lvl = classifyPM(d.pm25);
    buckets[lvl].push([d[axis], d.pm25]);
  });

  const option = {
    grid: { ...baseGrid, top: 36, bottom: 48 },
    tooltip: {
      ...baseTooltip,
      trigger: "item",
      formatter: (p: { value: [number, number]; seriesName: string }) =>
        `${p.seriesName}<br/>${xLabel}: <b>${p.value[0]}</b><br/>PM2.5: <b>${p.value[1]} µg/m³</b>`,
    },
    legend: {
      top: 0,
      textStyle: { color: ENVIRO_COLORS.axis, fontSize: 11 },
      icon: "circle",
    },
    xAxis: {
      type: "value",
      name: xLabel,
      nameLocation: "middle",
      nameGap: 28,
      nameTextStyle: baseTextStyle,
      axisLine,
      axisLabel: baseTextStyle,
      splitLine,
    },
    yAxis: {
      type: "value",
      name: "PM2.5 µg/m³",
      nameTextStyle: baseTextStyle,
      axisLine: { show: false },
      axisLabel: baseTextStyle,
      splitLine,
    },
    series: [
      {
        name: "Clean",
        type: "scatter",
        data: buckets.clean,
        symbolSize: 7,
        itemStyle: { color: ENVIRO_COLORS.clean, opacity: 0.75 },
      },
      {
        name: "Moderate",
        type: "scatter",
        data: buckets.moderate,
        symbolSize: 7,
        itemStyle: { color: ENVIRO_COLORS.moderate, opacity: 0.75 },
      },
      {
        name: "Poor",
        type: "scatter",
        data: buckets.poor,
        symbolSize: 7,
        itemStyle: { color: ENVIRO_COLORS.poor, opacity: 0.75 },
      },
    ],
  };

  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
