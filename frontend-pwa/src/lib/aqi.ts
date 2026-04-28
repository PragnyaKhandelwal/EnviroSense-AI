export type AqiLevel = "clean" | "moderate" | "poor";

export function classifyPM(value: number): AqiLevel {
  if (value < 15) return "clean";
  if (value <= 40) return "moderate";
  return "poor";
}

export function aqiLabel(level: AqiLevel): string {
  return level === "clean" ? "Clean Indoor Air" : level === "moderate" ? "Moderate Indoor Pollution" : "Poor Indoor Air";
}

export function aqiColorVar(level: AqiLevel): string {
  return level === "clean" ? "var(--clean)" : level === "moderate" ? "var(--moderate)" : "var(--poor)";
}

export function aqiTextClass(level: AqiLevel): string {
  return level === "clean" ? "text-clean" : level === "moderate" ? "text-moderate" : "text-poor";
}

export function aqiPanelClass(level: AqiLevel): string {
  return level === "clean" ? "panel-glow-clean" : level === "moderate" ? "panel-glow-moderate" : "panel-glow-poor";
}

export function aqiGlowText(level: AqiLevel): string {
  return level === "clean" ? "glow-text-clean" : level === "moderate" ? "glow-text-moderate" : "glow-text-poor";
}
