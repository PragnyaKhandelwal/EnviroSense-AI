/**
 * EnviroSense AI · Data Pipeline Layer
 * -----------------------------------
 * This module is the single source of truth for everything the UI renders
 * downstream of "live" data. In production this would be implemented by the
 * FastAPI/VPS backend; here it deterministically simulates the same shape so
 * the UI is bound to dynamic fields and can be wired to a real API by
 * swapping the body of `runPipeline()` with a `fetch()` call — no UI changes
 * needed.
 *
 * Pipeline stages (mirrors the real system):
 *   sensor_data   →  features  →  regime_profiles  →  regime_transitions  →  regime_stability
 *
 * The functions below are pure. Given the same `(deviceId, tick)` they always
 * return the same payload, which makes the dashboard snapshots stable and
 * easy to cache in localStorage for offline mode.
 */

import {
  detailedForecast,
  driftSeries,
  forecastData,
  initialAlerts,
  modelMetrics,
  regimeTransitions as staticTransitions,
  regimes,
  sparkline60,
  timeline24h,
  type AppAlert,
  type Regime,
} from "@/lib/mock-data";

const LIVE_API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() ?? "";

export const LIVE_API_ENABLED =
  ((import.meta.env.VITE_ENABLE_LIVE_API as string | undefined)?.toLowerCase() === "true") ||
  LIVE_API_BASE.length > 0;

// -----------------------------------------------------------------------------
// Devices — supports multiple sensors without UI changes.
// -----------------------------------------------------------------------------
export interface DeviceMeta {
  id: string;
  label: string;
  /** Per-device offset so each sensor produces a distinct readout. */
  seedOffset: number;
}

export const DEVICES: DeviceMeta[] = [
  { id: "terrace-01", label: "My Terrace-on-Room", seedOffset: 0 },
  { id: "kitchen-02", label: "Kitchen Probe", seedOffset: 7 },
  { id: "balcony-03", label: "Balcony Outdoor", seedOffset: 13 },
];

export const DEFAULT_DEVICE_ID = DEVICES[0].id;

export function getDevice(id: string): DeviceMeta {
  return DEVICES.find((d) => d.id === id) ?? DEVICES[0];
}

// -----------------------------------------------------------------------------
// Deterministic pseudo-random.
// -----------------------------------------------------------------------------
function rng(seed: number): () => number {
  return () => {
    const x = Math.sin(seed++ * 9301 + 49297) * 233280;
    return x - Math.floor(x);
  };
}

// -----------------------------------------------------------------------------
// 1) sensor_data — the live readings produced by a single device on the tick.
// -----------------------------------------------------------------------------
export interface SensorReading {
  device_id: string;
  device_label: string;
  ts: number;
  pm25: number;
  pm10: number;
  cityAvg: number;
  delta: number;
  temperature: number;
  humidity: number;
  /** Last 60 samples for the hero sparkline (µg/m³, oldest → newest). */
  sparkline: number[];
}

function buildSensorReading(device: DeviceMeta, tick: number): SensorReading {
  const r = rng(tick * 1000 + device.seedOffset * 17 + 1);
  const pm25 = +(8 + r() * 38).toFixed(1);
  const pm10 = +(pm25 + 6 + r() * 20).toFixed(1);
  return {
    device_id: device.id,
    device_label: device.label,
    ts: Date.now(),
    pm25,
    pm10,
    cityAvg: +(14 + r() * 26).toFixed(1),
    delta: +((r() * 4 - 2)).toFixed(1),
    temperature: +(20 + r() * 8).toFixed(1),
    humidity: +(45 + r() * 35).toFixed(1),
    // Reuse the deterministic 60-sample mock as the rolling window —
    // a real backend would slice the latest 60 readings here.
    sparkline: sparkline60,
  };
}

// -----------------------------------------------------------------------------
// 2) regime_profiles — classifier output for the current state.
// -----------------------------------------------------------------------------
export type RegimeLabel = Regime;

export interface RegimeProfile {
  current: RegimeLabel;
  confidence: number;
  /** Human-readable status summarising both regime and confidence. */
  statusLabel: "Stable Clean" | "Moderate Fluctuation" | "Unstable / Polluted";
  windowMinutes: number;
}

function classifyRegime(reading: SensorReading, tick: number): RegimeProfile {
  // Deterministic but PM-aware: the higher PM2.5 the more polluted the regime.
  const idx = Math.min(
    regimes.length - 1,
    Math.floor((reading.pm25 / 55) * regimes.length + (tick % 2)),
  );
  const current = regimes[idx];
  const r = rng(tick + 31);
  const confidence = +(0.7 + r() * 0.27).toFixed(2);

  const statusLabel: RegimeProfile["statusLabel"] =
    reading.pm25 < 15
      ? "Stable Clean"
      : reading.pm25 < 35
        ? "Moderate Fluctuation"
        : "Unstable / Polluted";

  return {
    current,
    confidence,
    statusLabel,
    windowMinutes: 5,
  };
}

// -----------------------------------------------------------------------------
// 3) regime_transitions — Markov-style next-state probabilities.
// -----------------------------------------------------------------------------
export interface RegimeTransitions {
  /** Square matrix [from][to]; rows sum to ~1. */
  matrix: number[][];
  /** Most likely next regime + its probability. */
  nextLikely: { regime: RegimeLabel; probability: number };
}

function buildTransitions(profile: RegimeProfile): RegimeTransitions {
  const fromIdx = regimes.indexOf(profile.current);
  const row = staticTransitions[fromIdx];
  // Pick the highest-probability target that is NOT the current regime.
  let bestIdx = -1;
  let bestProb = -Infinity;
  row.forEach((p, j) => {
    if (j !== fromIdx && p > bestProb) {
      bestProb = p;
      bestIdx = j;
    }
  });
  return {
    matrix: staticTransitions,
    nextLikely: {
      regime: regimes[bestIdx >= 0 ? bestIdx : 0],
      probability: bestProb >= 0 ? bestProb : 0,
    },
  };
}

// -----------------------------------------------------------------------------
// 4) regime_stability — duration in current state + entropy of next-state dist.
// -----------------------------------------------------------------------------
export interface RegimeStability {
  /** Minutes the system has remained in the current regime. */
  durationMinutes: number;
  /** Shannon entropy of the next-state distribution (0 = deterministic, ↑ = uncertain). */
  entropy: number;
  /** Compact label suitable for a UI badge. */
  trend: "Stable" | "Drifting" | "Volatile";
}

function buildStability(
  profile: RegimeProfile,
  transitions: RegimeTransitions,
  tick: number,
): RegimeStability {
  const fromIdx = regimes.indexOf(profile.current);
  const row = transitions.matrix[fromIdx];
  // Shannon entropy in nats.
  const entropy = +row
    .filter((p) => p > 0)
    .reduce((acc, p) => acc - p * Math.log(p), 0)
    .toFixed(2);
  const durationMinutes = 12 + ((tick * 7) % 90); // 12 → 102 min, deterministic
  const trend: RegimeStability["trend"] =
    entropy < 0.9 ? "Stable" : entropy < 1.3 ? "Drifting" : "Volatile";
  return { durationMinutes, entropy, trend };
}

// -----------------------------------------------------------------------------
// Latest anomaly (live-ish, jitters per tick) — derived from the alert feed.
// -----------------------------------------------------------------------------
export interface AnomalyState {
  anomalous: boolean;
  severity: "info" | "warning" | "critical";
  detectedAt: string;
  code: string;
  title: string;
}

function buildAnomalyState(alerts: AppAlert[]): AnomalyState {
  const open = alerts.find((a) => a.status !== "resolved") ?? alerts[0];
  if (!open) {
    return {
      anomalous: false,
      severity: "info",
      detectedAt: "—",
      code: "—",
      title: "All Clear",
    };
  }
  return {
    anomalous: open.severity !== "info",
    severity: open.severity,
    detectedAt: open.ts.split(" ")[1] ?? open.ts,
    code: open.id.replace("ALT-", "ANM-"),
    title: open.title,
  };
}

// -----------------------------------------------------------------------------
// Reliability / trust telemetry (small per-tick jitter, stable structure).
// -----------------------------------------------------------------------------
export interface ReliabilityState {
  trust: number;        // 0..100
  uptime: number;       // 0..100
  validity: number;     // 0..100
  driftSigma: number;   // σ
}

function buildReliability(tick: number, deviceOffset: number): ReliabilityState {
  const r = rng(tick + 99 + deviceOffset);
  const trust = Math.round(92 + r() * 7);            // 92–99
  const uptime = +(99 + r()).toFixed(1);              // 99.0–100.0
  const validity = +(96 + r() * 3.5).toFixed(1);     // 96–99.5
  const driftSigma = +(0.2 + r() * 0.8).toFixed(2);  // 0.2–1.0σ
  return { trust, uptime, validity, driftSigma };
}

// -----------------------------------------------------------------------------
// Pipeline output — what the snapshot actually carries.
// -----------------------------------------------------------------------------
export interface PipelineSnapshot {
  device: DeviceMeta;
  /** Stage 1 */
  sensor: SensorReading;
  /** Stage 2 */
  regime: RegimeProfile;
  /** Stage 3 */
  transitions: RegimeTransitions;
  /** Stage 4 */
  stability: RegimeStability;
  /** Derived signals consumed across the dashboard. */
  anomaly: AnomalyState;
  reliability: ReliabilityState;
  /** Live-ish references (refresh per tick, structure stable). */
  alerts: AppAlert[];
  drift: typeof driftSeries;
  modelMetrics: typeof modelMetrics;
  forecast: typeof forecastData;
  detailedForecast: typeof detailedForecast;
  timeline24h: typeof timeline24h;
}

function ensureSnapshotShape(payload: unknown): payload is PipelineSnapshot {
  const data = payload as Partial<PipelineSnapshot> | null;
  return Boolean(
    data &&
      data.sensor &&
      typeof data.sensor.ts === "number" &&
      Array.isArray(data.sensor.sparkline) &&
      data.regime &&
      data.transitions &&
      data.stability,
  );
}

/**
 * Fetches a live pipeline snapshot from the VPS API.
 * The API response shape mirrors `PipelineSnapshot` exactly.
 */
export async function fetchPipelineFromApi(
  deviceId: string,
  tick: number,
  timeoutMs = 4_000,
): Promise<PipelineSnapshot> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const base = LIVE_API_BASE.replace(/\/$/, "");
  const endpoint = `${base}/api/pipeline?device=${encodeURIComponent(deviceId)}&tick=${tick}`;

  try {
    const response = await fetch(endpoint, {
      method: "GET",
      signal: controller.signal,
      headers: {
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Pipeline API failed (${response.status})`);
    }

    const payload: unknown = await response.json();
    if (!ensureSnapshotShape(payload)) {
      throw new Error("Pipeline API returned an invalid payload shape");
    }
    return payload;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Run the full pipeline for a device at a given polling tick.
 * Replace the body with `await fetch('/api/pipeline?device=' + deviceId)`
 * to switch to the real backend — the SHAPE is identical.
 */
export function runPipeline(deviceId: string, tick: number): PipelineSnapshot {
  const device = getDevice(deviceId);
  const sensor = buildSensorReading(device, tick);
  const regime = classifyRegime(sensor, tick);
  const transitions = buildTransitions(regime);
  const stability = buildStability(regime, transitions, tick);
  const anomaly = buildAnomalyState(initialAlerts);
  const reliability = buildReliability(tick, device.seedOffset);

  return {
    device,
    sensor,
    regime,
    transitions,
    stability,
    anomaly,
    reliability,
    alerts: initialAlerts,
    drift: driftSeries,
    modelMetrics,
    forecast: forecastData,
    detailedForecast,
    timeline24h,
  };
}
