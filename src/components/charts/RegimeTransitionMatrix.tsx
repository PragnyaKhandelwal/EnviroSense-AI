import { EChart } from "@/components/charts/EChart";
import { ENVIRO_COLORS, baseTooltip } from "@/lib/echarts-theme";
import { useSensorData } from "@/lib/resilience";

export function RegimeTransitionMatrix({
  currentRegime,
  height = 320,
}: {
  currentRegime: string;
  height?: number;
}) {
  const { snapshot } = useSensorData();

  const matrix = snapshot?.transitions?.matrix || [];

  // fallback labels (match backend states)
  const regimes = ["Stable Clean", "Moderate", "Polluted"];

  const data: Array<[number, number, number]> = [];

  matrix.forEach((row: number[], i: number) => {
    row.forEach((val: number, j: number) => {
      data.push([j, i, val]);
    });
  });

  const currentIdx = regimes.indexOf(currentRegime);

  const option = {
    grid: { left: 110, right: 16, top: 28, bottom: 80 },
    tooltip: {
      ...baseTooltip,
      formatter: (p: any) => {
        const [to, from, v] = p.value;
        return `<b>${regimes[from]}</b> → <b>${regimes[to]}</b><br/>p = ${(v * 100).toFixed(0)}%`;
      },
    },
    xAxis: {
      type: "category",
      data: regimes,
      axisLabel: { color: ENVIRO_COLORS.axis, fontSize: 10 },
    },
    yAxis: {
      type: "category",
      data: regimes,
      inverse: true,
      axisLabel: {
        color: ENVIRO_COLORS.axis,
        formatter: (v: string) =>
          v === currentRegime ? `▶ ${v}` : v,
      },
    },
    visualMap: {
      min: 0,
      max: 1,
      orient: "horizontal",
      bottom: 0,
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
          formatter: (p: any) =>
            `${(p.value[2] * 100).toFixed(0)}%`,
        },
      },
    ],
  };

  return <EChart option={option} style={{ height, width: "100%" }} />;
}