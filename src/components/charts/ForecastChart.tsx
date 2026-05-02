import { useEffect, useState } from "react";
import { EChart } from "@/components/charts/EChart";
import {
  ENVIRO_COLORS,
  axisLine,
  baseGrid,
  baseTextStyle,
  baseTooltip,
  splitLine,
} from "@/lib/echarts-theme";

export function ForecastChart({ height = 360 }: { height?: number }) {
  const [history, setHistory] = useState<any[]>([]);
  const [forecast, setForecast] = useState<any[]>([]);

  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/api/forecast");
        const data = await res.json();

        if (!data || data.length === 0) return;

        // Split data (70% history, 30% forecast)
        const splitIndex = Math.floor(data.length * 0.7);

        const hist = data.slice(0, splitIndex).map((d: any, i: number, arr: any[]) => ({
          t: i - (arr.length - 1),
          value: d.y_true ?? d.y_pred,
        }));

        const fc = data.slice(splitIndex).map((d: any, i: number) => ({
          t: i + 1,
          value: d.y_pred,
          lower: d.y_pred - 2,
          upper: d.y_pred + 2,
        }));

        setHistory(hist);
        setForecast(fc);

      } catch (err) {
        console.error("Fetch error:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 3000); // live refresh

    return () => clearInterval(interval);
  }, []);

  // -----------------------------
  // 📊 Chart Data Processing
  // -----------------------------

  const xData = [
    ...history.map((p) => p.t.toString()),
    ...forecast.map((p) => p.t.toString()),
  ];

  const histSeries = [
    ...history.map((p) => p.value),
    ...forecast.map(() => null),
  ];

  const fcSeries = [
    ...history.map(() => null),
    ...forecast.map((p) => p.value),
  ];

  // connect last point
  if (history.length > 0) {
    fcSeries[history.length - 1] = history[history.length - 1].value;
  }

  const lower = [
    ...history.map(() => null),
    ...forecast.map((p) => p.lower),
  ];

  const upper = [
    ...history.map(() => null),
    ...forecast.map((p) => p.upper - p.lower),
  ];

  if (history.length > 0) {
    lower[history.length - 1] = history[history.length - 1].value;
    upper[history.length - 1] = 0;
  }

  // -----------------------------
  // 📈 Chart Config
  // -----------------------------

  const option = {
    grid: { ...baseGrid, top: 32, bottom: 36 },

    tooltip: {
      trigger: "axis",
      ...baseTooltip,
      formatter: (params: any) => {
        const t = +params[0].axisValue;
        const tag =
          t < 0 ? `${t} min (history)` :
          t === 0 ? "Now" :
          `+${t} min (forecast)`;

        const lines = params
          .filter((p: any) => p.seriesName === "History" || p.seriesName === "Forecast")
          .map(
            (p: any) =>
              `<div style="display:flex;justify-content:space-between">
                <span style="color:${p.color}">● ${p.seriesName}</span>
                <b>${p.value?.toFixed?.(2) ?? "—"} µg/m³</b>
              </div>`
          )
          .join("");

        return `<div>${tag}</div>${lines}`;
      },
    },

    legend: {
      data: ["History", "Forecast", "95% CI"],
      textStyle: { color: ENVIRO_COLORS.axis, fontSize: 11 },
      top: 0,
      right: 8,
    },

    xAxis: {
      type: "category",
      data: xData,
      axisLine,
      axisLabel: baseTextStyle,
      splitLine: { show: false },
    },

    yAxis: {
      type: "value",
      name: "PM2.5 µg/m³",
      nameTextStyle: baseTextStyle,
      axisLabel: baseTextStyle,
      splitLine,
    },

    series: [
      {
        name: "ci-base",
        type: "line",
        data: lower,
        stack: "ci",
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { color: "transparent" },
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
      },
      {
        name: "Forecast",
        type: "line",
        data: fcSeries,
        smooth: true,
        symbol: "none",
        lineStyle: { color: ENVIRO_COLORS.cyan, width: 2, type: "dashed" },
      },
    ],
  };

  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
