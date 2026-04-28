import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  DEFAULT_DEVICE_ID,
  DEVICES,
  LIVE_API_ENABLED,
  fetchPipelineFromApi,
  runPipeline,
  type DeviceMeta,
  type PipelineSnapshot,
} from "@/lib/pipeline";

/**
 * The "snapshot" is the full pipeline output for the active device.
 * Everything the dashboard renders downstream of the polling layer reads
 * from here — no component imports raw mocks for live data anymore.
 *
 *   sensor_data   → snapshot.sensor
 *   regime_profiles    → snapshot.regime
 *   regime_transitions → snapshot.transitions
 *   regime_stability   → snapshot.stability
 */
export type SensorSnapshot = PipelineSnapshot;

const STORAGE_KEY_PREFIX = "envirosense.snapshot.v2";
const DEVICE_KEY = "envirosense.device.v1";
const POLL_INTERVAL_MS = 60_000;
const FETCH_TIMEOUT_MS = 4_000;

function storageKeyFor(deviceId: string) {
  return `${STORAGE_KEY_PREFIX}.${deviceId}`;
}

/**
 * Simulated micro-batch fetch. In production this would hit FastAPI; here it
 * resolves with the deterministic pipeline output for the active device OR
 * rejects when the developer outage toggle is set.
 */
function simulateFetch(
  deviceId: string,
  tick: number,
  forceFail: boolean,
): Promise<SensorSnapshot> {
  if (forceFail) {
    return Promise.reject(new Error("Simulated server outage (FastAPI unreachable)"));
  }

  if (LIVE_API_ENABLED) {
    return fetchPipelineFromApi(deviceId, tick, FETCH_TIMEOUT_MS);
  }

  return new Promise((resolve, reject) => {
    const latency = 180 + Math.random() * 220;
    const timer = setTimeout(() => {
      resolve(runPipeline(deviceId, tick));
    }, latency);
    setTimeout(() => {
      clearTimeout(timer);
      reject(new Error("Request timed out"));
    }, FETCH_TIMEOUT_MS);
  });
}

function readCache(deviceId: string): SensorSnapshot | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(storageKeyFor(deviceId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as SensorSnapshot;
    // Minimal shape validation — keeps the UI safe from corrupted cache.
    if (
      typeof parsed?.sensor?.ts !== "number" ||
      !Array.isArray(parsed?.sensor?.sparkline) ||
      !parsed?.regime ||
      !parsed?.transitions ||
      !parsed?.stability
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function writeCache(snap: SensorSnapshot) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      storageKeyFor(snap.device.id),
      JSON.stringify(snap),
    );
  } catch {
    /* quota or private mode — silently ignore */
  }
}

function readStoredDeviceId(): string {
  if (typeof window === "undefined") return DEFAULT_DEVICE_ID;
  try {
    const raw = window.localStorage.getItem(DEVICE_KEY);
    if (raw && DEVICES.some((d) => d.id === raw)) return raw;
  } catch {
    /* ignore */
  }
  return DEFAULT_DEVICE_ID;
}

interface ResilienceContextValue {
  snapshot: SensorSnapshot;
  isLive: boolean;
  lastSyncTs: number | null;
  failureReason: string | null;
  outageSimulated: boolean;
  setOutageSimulated: (v: boolean) => void;
  /** Multi-device support — UI can switch active sensor without re-mount. */
  deviceId: string;
  devices: DeviceMeta[];
  setDeviceId: (id: string) => void;
  /** Force a refetch immediately. */
  refresh: () => void;
}

const ResilienceContext = createContext<ResilienceContextValue | null>(null);

export function ResilienceProvider({ children }: { children: ReactNode }) {
  // Bootstrap snapshot from a deterministic seed (SSR-safe).
  const [deviceId, setDeviceIdState] = useState<string>(DEFAULT_DEVICE_ID);
  const [snapshot, setSnapshot] = useState<SensorSnapshot>(() =>
    runPipeline(DEFAULT_DEVICE_ID, 0),
  );
  const [isLive, setIsLive] = useState(true);
  const [lastSyncTs, setLastSyncTs] = useState<number | null>(null);
  const [failureReason, setFailureReason] = useState<string | null>(null);
  const [outageSimulated, setOutageSimulated] = useState(false);
  const tickRef = useRef(0);
  const outageRef = useRef(false);
  outageRef.current = outageSimulated;

  // Hydrate device + cache after mount (avoids SSR mismatch).
  useEffect(() => {
    const stored = readStoredDeviceId();
    if (stored !== deviceId) setDeviceIdState(stored);
    const cached = readCache(stored);
    if (cached) {
      setSnapshot(cached);
      setLastSyncTs(cached.sensor.ts);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const doFetch = useCallback(
    async (targetDeviceId: string) => {
      tickRef.current += 1;
      try {
        const next = await simulateFetch(
          targetDeviceId,
          tickRef.current,
          outageRef.current,
        );
        setSnapshot(next);
        setIsLive(true);
        setLastSyncTs(next.sensor.ts);
        setFailureReason(null);
        writeCache(next);
      } catch (err) {
        const reason = err instanceof Error ? err.message : "Unknown fetch error";
        setFailureReason(reason);
        setIsLive(false);
        // Keep last-known snapshot on screen — graceful degradation.
        const cached = readCache(targetDeviceId);
        if (cached) setSnapshot(cached);
      }
    },
    [],
  );

  // Initial fetch + 60s polling loop, restarts when active device changes.
  useEffect(() => {
    doFetch(deviceId);
    const id = setInterval(() => doFetch(deviceId), POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [deviceId, doFetch]);

  // When the dev outage toggle flips, immediately revalidate.
  useEffect(() => {
    doFetch(deviceId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [outageSimulated]);

  const setDeviceId = useCallback((id: string) => {
    if (!DEVICES.some((d) => d.id === id)) return;
    setDeviceIdState(id);
    if (typeof window !== "undefined") {
      try {
        window.localStorage.setItem(DEVICE_KEY, id);
      } catch {
        /* ignore */
      }
    }
  }, []);

  return (
    <ResilienceContext.Provider
      value={{
        snapshot,
        isLive,
        lastSyncTs,
        failureReason,
        outageSimulated,
        setOutageSimulated,
        deviceId,
        devices: DEVICES,
        setDeviceId,
        refresh: () => doFetch(deviceId),
      }}
    >
      {children}
    </ResilienceContext.Provider>
  );
}

export function useSensorData() {
  const ctx = useContext(ResilienceContext);
  if (!ctx) {
    // Provider not mounted (e.g. tests, SSR shell) — return a static snapshot
    // so consumers never crash. This is part of "graceful degradation".
    return {
      snapshot: runPipeline(DEFAULT_DEVICE_ID, 0),
      isLive: true,
      lastSyncTs: null,
      failureReason: null,
      outageSimulated: false,
      setOutageSimulated: () => {},
      deviceId: DEFAULT_DEVICE_ID,
      devices: DEVICES,
      setDeviceId: () => {},
      refresh: () => {},
    } satisfies ResilienceContextValue;
  }
  return ctx;
}

/** Format helper — "12s ago", "4m ago", "—" */
export function formatRelative(ts: number | null): string {
  if (!ts) return "—";
  const diff = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}
