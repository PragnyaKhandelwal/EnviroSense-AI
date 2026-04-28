// Deterministic pseudo-random for stable mock data
function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(42);

// Forecast: Next 30 minutes PM2.5 with confidence interval
export const forecastData = Array.from({ length: 31 }, (_, i) => {
  const base = 18 + Math.sin(i / 5) * 6 + rand() * 3;
  const ci = 3 + i * 0.18;
  return {
    minute: i,
    pm25: +base.toFixed(1),
    lower: +(base - ci).toFixed(1),
    upper: +(base + ci).toFixed(1),
    band: [+(base - ci).toFixed(1), +(base + ci).toFixed(1)] as [number, number],
  };
});

// Detailed predictive forecast: 60 min history (solid) + 60 min forecast (dashed) with CI band
export const detailedForecast = (() => {
  const r = mulberry32(101);
  const history = Array.from({ length: 60 }, (_, i) => {
    const t = i - 60; // -60..-1 minutes
    const base = 22 + Math.sin(i / 7) * 5 + (i / 60) * 4 + r() * 2.5;
    return { t, value: +base.toFixed(1) };
  });
  const last = history[history.length - 1].value;
  const forecast = Array.from({ length: 60 }, (_, i) => {
    const t = i + 1;
    const trend = last + Math.sin((i + 5) / 8) * 4 + i * 0.06 + r() * 1.5;
    const ci = 2 + i * 0.12;
    return {
      t,
      value: +trend.toFixed(1),
      lower: +(trend - ci).toFixed(1),
      upper: +(trend + ci).toFixed(1),
    };
  });
  return { history, forecast };
})();

// 24h timeline: PM2.5 overlaid with temperature & humidity
export const timeline24h = Array.from({ length: 24 }, (_, h) => {
  const morning = Math.exp(-((h - 8) ** 2) / 6) * 18;
  const evening = Math.exp(-((h - 19) ** 2) / 8) * 22;
  const pm25 = +(10 + morning + evening + (rand() - 0.5) * 4).toFixed(1);
  const temp = +(20 + Math.sin(((h - 6) / 24) * Math.PI * 2) * 6 + rand()).toFixed(1);
  const humidity = +(55 - Math.sin(((h - 6) / 24) * Math.PI * 2) * 18 + rand() * 4).toFixed(1);
  return { hour: h, pm25, temp, humidity };
});

// 60-minute sparkline for hero card
export const sparkline60 = Array.from({ length: 60 }, (_, i) => {
  const v = 18 + Math.sin(i / 6) * 5 + Math.cos(i / 11) * 3 + rand() * 2;
  return +v.toFixed(1);
});

// Weekly Diurnal Heatmap: 7 days x 24 hours
const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
function makeHeatmap(metric: "pm25" | "pm10") {
  const scale = metric === "pm10" ? 1.7 : 1;
  return days.map((day, di) => ({
    day,
    hours: Array.from({ length: 24 }, (_, h) => {
      const morningPeak = Math.exp(-((h - 8) ** 2) / 6) * 28;
      const eveningPeak = Math.exp(-((h - 19) ** 2) / 8) * 34;
      const weekendShift = di >= 5 ? -6 : 0;
      const noise = (rand() - 0.5) * 6;
      const value = Math.max(2, (8 + morningPeak + eveningPeak + weekendShift + noise) * scale);
      return { hour: h, value: +value.toFixed(1) };
    }),
  }));
}
export const heatmapPM25 = makeHeatmap("pm25");
export const heatmapPM10 = makeHeatmap("pm10");

// Particle size distribution (stacked area) per scenario
const bins = ["0.3-1µm", "1-2.5µm", "2.5-5µm", "5-10µm", ">10µm"];
function scenario(scale: number[]) {
  return Array.from({ length: 24 }, (_, t) => {
    const row: Record<string, number | string> = { t };
    bins.forEach((b, i) => {
      const wave = Math.sin((t + i * 4) / 6) * 8 + 20;
      row[b] = Math.max(2, +(wave * scale[i] + rand() * 4).toFixed(1));
    });
    return row;
  });
}
export const sizeScenarios = {
  Normal: scenario([1, 1, 1, 1, 1]),
  "Dust Storm": scenario([0.4, 0.6, 1.4, 2.2, 3.0]),
  Rain: scenario([0.3, 0.4, 0.5, 0.4, 0.3]),
  "Post-Rain Clean Air": scenario([0.5, 0.5, 0.4, 0.3, 0.2]),
};
export const sizeBins = bins;

// Density scatter: count vs mass for bins
export const densityScatter = bins.map((bin, idx) => ({
  bin,
  points: Array.from({ length: 40 }, () => ({
    count: +(50 + rand() * 400 * (idx + 1)).toFixed(1),
    mass: +(2 + rand() * 30 * (idx + 1) ** 1.4).toFixed(1),
  })),
}));

// Correlation scatter: PM2.5 vs Humidity vs Temp
export const correlationData = Array.from({ length: 220 }, () => {
  const humidity = 30 + rand() * 60;
  const temp = 18 + rand() * 18;
  const pm25 = Math.max(2, 12 + (humidity - 50) * 0.25 + (35 - temp) * 0.5 + rand() * 10);
  return { humidity: +humidity.toFixed(1), temp: +temp.toFixed(1), pm25: +pm25.toFixed(1) };
});

// Anomalies log
export const anomalies = [
  { ts: "2025-04-18 14:32:11", label: "Sudden Spike — Isolation Forest", level: "poor", code: "ANM-204" },
  { ts: "2025-04-18 13:18:44", label: "Sensor Drift Detected — DBSCAN", level: "moderate", code: "ANM-118" },
  { ts: "2025-04-18 11:02:09", label: "Brief Dropout — Watchdog", level: "moderate", code: "WRN-007" },
  { ts: "2025-04-18 09:47:30", label: "Calibration Cluster Shift", level: "moderate", code: "ANM-091" },
  { ts: "2025-04-18 07:12:54", label: "Restored to Normal Range", level: "clean", code: "OK-200" },
  { ts: "2025-04-17 22:51:03", label: "Spike — Cooking Event Inferred", level: "poor", code: "ANM-187" },
  { ts: "2025-04-17 18:09:22", label: "Humidity Cross-Correlation Outlier", level: "moderate", code: "ANM-142" },
] as const;

// Device comparison (Sensor 1 vs Sensor 2) — boxplot stats
export const deviceStats = [
  { hour: "00", s1: { min: 6, q1: 9, med: 12, q3: 16, max: 22 }, s2: { min: 7, q1: 10, med: 13, q3: 17, max: 24 } },
  { hour: "04", s1: { min: 5, q1: 8, med: 11, q3: 14, max: 19 }, s2: { min: 5, q1: 8, med: 11, q3: 14, max: 20 } },
  { hour: "08", s1: { min: 12, q1: 18, med: 26, q3: 34, max: 48 }, s2: { min: 13, q1: 19, med: 27, q3: 36, max: 51 } },
  { hour: "12", s1: { min: 10, q1: 14, med: 20, q3: 28, max: 38 }, s2: { min: 11, q1: 15, med: 21, q3: 29, max: 41 } },
  { hour: "16", s1: { min: 11, q1: 16, med: 22, q3: 30, max: 42 }, s2: { min: 12, q1: 17, med: 23, q3: 31, max: 44 } },
  { hour: "20", s1: { min: 14, q1: 22, med: 32, q3: 42, max: 58 }, s2: { min: 15, q1: 23, med: 33, q3: 44, max: 60 } },
];

// ----- New mock data for the AI pipeline -----

export type Regime =
  | "Post-Rain Clean"
  | "Stable Indoor"
  | "Traffic Spike"
  | "Cooking Event"
  | "Dust Influx";

export const regimes: Regime[] = [
  "Post-Rain Clean",
  "Stable Indoor",
  "Traffic Spike",
  "Cooking Event",
  "Dust Influx",
];

// Transition matrix (rows: from, cols: to). Probabilities, rows sum ~= 1
export const regimeTransitions: number[][] = [
  [0.55, 0.30, 0.05, 0.05, 0.05],
  [0.10, 0.62, 0.14, 0.10, 0.04],
  [0.04, 0.18, 0.55, 0.15, 0.08],
  [0.06, 0.22, 0.10, 0.55, 0.07],
  [0.05, 0.20, 0.20, 0.05, 0.50],
];

// Drift control chart: rolling z-score over 60 samples
export const driftSeries = Array.from({ length: 60 }, (_, i) => {
  const drift = i > 40 ? (i - 40) * 0.08 : 0;
  return +(Math.sin(i / 6) * 0.6 + (rand() - 0.5) * 0.7 + drift).toFixed(2);
});

// Rolling MAE / RMSE for the last N evaluation windows
export const modelMetrics = Array.from({ length: 24 }, (_, i) => ({
  t: i,
  mae: +(2.4 + Math.sin(i / 5) * 0.4 + rand() * 0.3).toFixed(2),
  rmse: +(3.6 + Math.sin(i / 5) * 0.5 + rand() * 0.4).toFixed(2),
}));

export const modelRegistry = [
  { name: "PM2.5 Forecaster", version: "XGBoost v2.1", mae: 2.31, rmse: 3.42, status: "active" },
  { name: "Anomaly Detector", version: "IsolationForest v1.4", mae: 0.12, rmse: 0.18, status: "active" },
  { name: "Regime Classifier", version: "HMM v0.9", mae: 0.08, rmse: 0.14, status: "shadow" },
  { name: "Trust Scorer", version: "Bayesian v1.0", mae: 0.04, rmse: 0.07, status: "active" },
];

// Live alerts feed
export type AlertSeverity = "info" | "warning" | "critical";
export type AlertStatus = "open" | "acknowledged" | "resolved";
export interface AppAlert {
  id: string;
  ts: string;
  severity: AlertSeverity;
  title: string;
  description: string;
  recommendation?: string;
  status: AlertStatus;
}

export const initialAlerts: AppAlert[] = [
  {
    id: "ALT-2041",
    ts: "2025-04-18 14:32:11",
    severity: "critical",
    title: "Sudden PM2.5 Anomaly + Low Reliability",
    description: "Isolation Forest flagged a 3.4σ deviation while Trust Score dropped to 71%.",
    recommendation:
      "Check sensor for physical obstruction; drift detected. Suggest field calibration within 24h.",
    status: "open",
  },
  {
    id: "ALT-2040",
    ts: "2025-04-18 13:18:44",
    severity: "warning",
    title: "Sensor Drift Detected",
    description: "DBSCAN cluster shift across the last 30 minute window.",
    recommendation: "Schedule baseline re-calibration on next maintenance cycle.",
    status: "open",
  },
  {
    id: "ALT-2039",
    ts: "2025-04-18 11:02:09",
    severity: "info",
    title: "Brief Network Dropout",
    description: "Watchdog observed 11s gap; auto-recovered, no data loss.",
    status: "acknowledged",
  },
  {
    id: "ALT-2038",
    ts: "2025-04-18 09:47:30",
    severity: "warning",
    title: "Regime Transition: Stable → Traffic Spike",
    description: "HMM regime classifier signalled a confident transition (p=0.81).",
    recommendation: "Consider closing windows during 8–10am and 6–8pm peaks.",
    status: "open",
  },
  {
    id: "ALT-2037",
    ts: "2025-04-17 22:51:03",
    severity: "critical",
    title: "Cooking Event Inferred",
    description: "Particle bin shift consistent with indoor combustion source.",
    recommendation: "Increase ventilation; HEPA cycle for next 30 minutes.",
    status: "resolved",
  },
];
