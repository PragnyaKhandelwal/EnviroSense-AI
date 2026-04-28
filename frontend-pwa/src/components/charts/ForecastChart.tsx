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

/**
 * Detailed forecast chart used on Predictive Intelligence:
 *  - Solid historical line for past 60 minutes
 *  - "Now" vertical reference line at t=0
 *  - Dashed predicted line for next 60 minutes
 *  - Shaded confidence band around the forecast
 */
export function ForecastChart({ height = 360 }: { height?: number }) {
  const { snapshot } = useSensorData();
  const { history, forecast } = snapshot.detailedForecast;

  const xData = [
    ...history.map((p) => p.t.toString()),
    ...forecast.map((p) => p.t.toString()),
  ];

  const histSeries: (number | null)[] = [
    ...history.map((p) => p.value),
    ...forecast.map(() => null),
  ];
  const fcSeries: (number | null)[] = [
    ...history.map(() => null),
    ...forecast.map((p) => p.value),
  ];
  // bridge last historical point so dashed line connects visually
  fcSeries[history.length - 1] = history[history.length - 1].value;

  const lower: (number | null)[] = [
    ...history.map(() => null),
    ...forecast.map((p) => p.lower),
  ];
  const upper: (number | null)[] = [
    ...history.map(() => null),
    ...forecast.map((p) => p.upper - p.lower), // stacked: band = upper - lower
  ];
  lower[history.length - 1] = history[history.length - 1].value;
  upper[history.length - 1] = 0;

  const option = {
    grid: { ...baseGrid, top: 32, bottom: 36 },
    tooltip: {
      trigger: "axis",
      ...baseTooltip,
      formatter: (params: Array<{ axisValue: string; seriesName: string; value: number; color: string }>) => {
        const t = +params[0].axisValue;
        const tag = t < 0 ? `${t} min (history)` : t === 0 ? "Now" : `+${t} min (forecast)`;
        const lines = params
          .filter((p) => p.seriesName === "History" || p.seriesName === "Forecast")
          .map(
            (p) =>
              `<div style="display:flex;justify-content:space-between;gap:12px"><span style="color:${p.color}">● ${p.seriesName}</span><b>${p.value?.toFixed?.(1) ?? "—"} µg/m³</b></div>`,
          )
          .join("");
        return `<div style="font-size:11px;color:${ENVIRO_COLORS.axis};margin-bottom:4px">${tag}</div>${lines}`;
      },
    },
    legend: {
      data: ["History", "Forecast", "95% CI"],
      textStyle: { color: ENVIRO_COLORS.axis, fontSize: 11 },
      top: 0, right: 8,
      icon: "roundRect",
    },
    xAxis: {
      type: "category",
      data: xData,
      axisLine,
      axisLabel: {
        ...baseTextStyle,
        formatter: (v: string) => {
          const n = +v;
          if (n === -60) return "−60m";
          if (n === 0) return "Now";
          if (n === 60) return "+60m";
          return n % 15 === 0 ? (n > 0 ? `+${n}` : `${n}`) : "";
        },
      },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      name: "PM2.5 µg/m³",
      nameTextStyle: { ...baseTextStyle, padding: [0, 0, 0, 30] },
      axisLine: { show: false },
      axisLabel: baseTextStyle,
      splitLine,
    },
    series: [
      // CI band: stacked area lower (transparent) + band (translucent)
      {
        name: "ci-base",
        type: "line",
        data: lower,
        stack: "ci",
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { color: "transparent" },
        tooltip: { show: false },
        silent: true,
      },
      {
        name: "95% CI",
        type: "line",
        data: upper,
        stack: "ci",
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { color: "oklch(0.80 0.14 200 / 0.18)" },
      },
      {
        name: "History",
        type: "line",
        data: histSeries,
        smooth: true,
        symbol: "none",
        lineStyle: { color: ENVIRO_COLORS.clean, width: 2 },
        connectNulls: false,
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: ENVIRO_COLORS.cyan, type: "dashed", width: 1.4 },
          label: {
            formatter: "Now",
            color: ENVIRO_COLORS.cyan,
            fontSize: 10,
          },
          data: [{ xAxis: "0" }],
        },
      },
      {
        name: "Forecast",
        type: "line",
        data: fcSeries,
        smooth: true,
        symbol: "none",
        lineStyle: { color: ENVIRO_COLORS.cyan, width: 2, type: "dashed" },
        connectNulls: false,
      },
    ],
  };

  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
