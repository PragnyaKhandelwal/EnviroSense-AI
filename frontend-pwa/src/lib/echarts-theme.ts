// Shared ECharts options aligned with the design system.
// Colors are read from CSS variables so theming stays consistent.

export const ENVIRO_COLORS = {
  clean: "oklch(0.78 0.18 150)",
  moderate: "oklch(0.85 0.16 90)",
  poor: "oklch(0.68 0.24 25)",
  cyan: "oklch(0.80 0.14 200)",
  violet: "oklch(0.72 0.18 290)",
  amber: "oklch(0.78 0.16 50)",
  grid: "oklch(0.30 0.02 240 / 0.4)",
  axis: "oklch(0.70 0.02 240)",
  bg: "oklch(0.20 0.025 245)",
  border: "oklch(0.30 0.02 240)",
  text: "oklch(0.96 0.01 230)",
} as const;

export const baseTextStyle = {
  fontFamily:
    "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
  fontSize: 11,
  color: ENVIRO_COLORS.axis,
};

export const baseTooltip = {
  backgroundColor: ENVIRO_COLORS.bg,
  borderColor: ENVIRO_COLORS.border,
  borderWidth: 1,
  textStyle: { color: ENVIRO_COLORS.text, fontSize: 12 },
  extraCssText:
    "border-radius: 8px; box-shadow: 0 8px 24px -8px rgba(0,0,0,.5); backdrop-filter: blur(8px);",
};

export const baseGrid = {
  left: 36,
  right: 16,
  top: 24,
  bottom: 28,
  containLabel: true,
};

export const axisLine = {
  lineStyle: { color: ENVIRO_COLORS.border },
};
export const splitLine = {
  lineStyle: { color: ENVIRO_COLORS.grid, type: "dashed" as const },
};

export function levelColor(level: "clean" | "moderate" | "poor") {
  return level === "clean"
    ? ENVIRO_COLORS.clean
    : level === "moderate"
      ? ENVIRO_COLORS.moderate
      : ENVIRO_COLORS.poor;
}
