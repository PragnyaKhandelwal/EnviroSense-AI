import { EChart } from "@/components/charts/EChart";
import { ENVIRO_COLORS } from "@/lib/echarts-theme";

/** Circular trust gauge (0-100). Color shifts with confidence. */
export function TrustGauge({
  value,
  height = 200,
  label = "Trust Score",
}: {
  value: number;
  height?: number;
  label?: string;
}) {
  const color =
    value >= 90 ? ENVIRO_COLORS.clean : value >= 70 ? ENVIRO_COLORS.moderate : ENVIRO_COLORS.poor;

  const option = {
    series: [
      {
        type: "gauge",
        startAngle: 220,
        endAngle: -40,
        min: 0,
        max: 100,
        radius: "92%",
        progress: {
          show: true,
          width: 12,
          itemStyle: { color, shadowColor: color, shadowBlur: 12 },
        },
        axisLine: {
          lineStyle: {
            width: 12,
            color: [[1, ENVIRO_COLORS.border]],
          },
        },
        pointer: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        anchor: { show: false },
        title: { show: false },
        detail: {
          valueAnimation: true,
          fontSize: 28,
          fontWeight: 700,
          color,
          offsetCenter: [0, "5%"],
          formatter: (v: number) => `${v.toFixed(0)}%`,
        },
        data: [{ value }],
      },
    ],
  };

  return (
    <div className="relative">
      <EChart option={option} style={{ height, width: "100%" }} notMerge />
      <div className="absolute inset-x-0 bottom-2 text-center text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
        {label}
      </div>
    </div>
  );
}
