import { EChart } from "@/components/charts/EChart";
import {
  ENVIRO_COLORS,
  baseTooltip,
} from "@/lib/echarts-theme";
import { regimes, regimeTransitions, type Regime } from "@/lib/mock-data";

/** Heatmap-style transition matrix from current regime to next-step probabilities. */
export function RegimeTransitionMatrix({
  currentRegime,
  height = 320,
}: {
  currentRegime: Regime;
  height?: number;
}) {
  const data: Array<[number, number, number]> = [];
  regimes.forEach((_from, i) => {
    regimes.forEach((_to, j) => {
      data.push([j, i, +regimeTransitions[i][j].toFixed(2)]);
    });
  });

  const currentIdx = regimes.indexOf(currentRegime);

  const option = {
    grid: { left: 110, right: 16, top: 28, bottom: 80, containLabel: false },
    tooltip: {
      ...baseTooltip,
      formatter: (p: { value: [number, number, number] }) => {
        const [to, from, v] = p.value;
        return `<b>${regimes[from]}</b> → <b>${regimes[to]}</b><br/>p = ${(v * 100).toFixed(0)}%`;
      },
    },
    xAxis: {
      type: "category",
      data: regimes,
      position: "bottom",
      splitArea: { show: false },
      axisLine: { lineStyle: { color: ENVIRO_COLORS.border } },
      axisLabel: {
        color: ENVIRO_COLORS.axis,
        fontSize: 10,
        rotate: 30,
        interval: 0,
      },
      name: "to →",
      nameTextStyle: { color: ENVIRO_COLORS.axis, fontSize: 10 },
      nameLocation: "end",
    },
    yAxis: {
      type: "category",
      data: regimes,
      inverse: true,
      axisLine: { lineStyle: { color: ENVIRO_COLORS.border } },
      axisLabel: {
        color: ENVIRO_COLORS.axis,
        fontSize: 10,
        formatter: (v: string) => (v === currentRegime ? `▶ ${v}` : v),
      },
    },
    visualMap: {
      min: 0,
      max: 1,
      calculable: false,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      itemWidth: 10,
      itemHeight: 100,
      textStyle: { color: ENVIRO_COLORS.axis, fontSize: 10 },
      inRange: {
        color: [
          "oklch(0.20 0.025 245)",
          "oklch(0.40 0.10 200)",
          ENVIRO_COLORS.cyan,
          ENVIRO_COLORS.clean,
        ],
      },
    },
    series: [
      {
        type: "heatmap",
        data,
        label: {
          show: true,
          color: ENVIRO_COLORS.text,
          fontSize: 10,
          formatter: (p: { value: [number, number, number] }) =>
            `${(p.value[2] * 100).toFixed(0)}%`,
        },
        emphasis: { itemStyle: { borderColor: ENVIRO_COLORS.clean, borderWidth: 1.5 } },
        itemStyle: { borderColor: ENVIRO_COLORS.bg, borderWidth: 1 },
        markArea: {
          silent: true,
          itemStyle: { color: "transparent", borderColor: ENVIRO_COLORS.clean, borderWidth: 1.5 },
          data: [
            [
              { yAxis: regimes[currentIdx], xAxis: regimes[0] },
              { yAxis: regimes[currentIdx], xAxis: regimes[regimes.length - 1] },
            ],
          ],
        },
      },
    ],
  };

  return <EChart option={option} style={{ height, width: "100%" }} notMerge />;
}
