import { createContext, useContext, useEffect, useState } from "react";

type Snapshot = {
  timeline24h: {
    hour: number;
    pm25: number;
    temperature: number;
    humidity: number;
  }[];

  modelMetrics: any[];
  drift: any[];
  forecast: any[];
  detailedForecast: any[];

  // ✅ member3 data
  particleSize: any[];
  densityScatter: any[];
  clusters: any[];

  alerts: any[];

  device?: any;
  sensor?: any;
  regime?: {
    current: string;
    confidence: number;
  };
  transitions?: {
    matrix: number[][];
    labels?: string[];
    currentIndex?: number;
    nextLikely?: {
      index: number;
      regime: string;
      probability: number;
    };
  };

  reliability?: {
    trust: number;
    uptime: number;
    validity: number;
    driftSigma?: number;
  };
  anomaly?: any;
};

const ResilienceContext = createContext<{
  snapshot: Snapshot | null;
  loading: boolean;
  error: string | null;
}>({
  snapshot: null,
  loading: true,
  error: null,
});

export function ResilienceProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const normalize = (arr: any[]) =>
    Array.isArray(arr)
      ? arr.map((d: any) => ({
          ...d,
          // ✅ unify all possible backend formats
          pm25: d.pm25 ?? d.pm2_5 ?? 0,
          pm10: d.pm10 ?? d.pm10_0 ?? 0,
        }))
      : [];
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [pipelineRes, physicsRes, regimeRes] = await Promise.all([
          fetch("/api/pipeline"),
          fetch("/api/particle-physics").catch(() => null),
          fetch("/api/regime-intelligence").catch(() => null),
        ]);

        const pipeline = await pipelineRes.json();

        // ✅ correct fallback keys (FIXED)
        let physics = {
          particleSize: [],
          densityScatter: [],
          clusters: [],
        };

        if (physicsRes && physicsRes.ok) {
          const data = await physicsRes.json();

          physics = {
            particleSize: Array.isArray(data.particleSize)
              ? data.particleSize
              : [],
            densityScatter: Array.isArray(data.densityScatter)
              ? data.densityScatter
              : [],
            clusters: Array.isArray(data.clusters)
              ? data.clusters
              : [],
          };
        } else {
          console.warn("⚠️ particle-physics API missing, using empty fallback");
        }
        let transitions = {
          matrix: [],
          labels: ["Stable Clean", "Moderate", "Polluted"],
          currentIndex: 0,
        };
        
        if (regimeRes && regimeRes.ok) {
          const regimeData = await regimeRes.json();
        
          transitions = {
            matrix: regimeData?.transitions?.matrix || [],
            labels: regimeData?.transitions?.labels || [
              "Stable Clean",
              "Moderate",
              "Polluted",
            ],
            currentIndex: regimeData?.transitions?.currentIndex ?? 0,
          };
        } else {
          console.warn("⚠️ regime-intelligence API missing");
        }

        const safeData: Snapshot = {
          timeline24h: Array.isArray(pipeline.timeline24h)
            ? pipeline.timeline24h.map((d: any) => ({
                hour: d.hour ?? 0,
                pm25: d.pm25 ?? 0,
                temperature: d.temperature ?? d.temp ?? 0,
                humidity: d.humidity ?? 0,
              }))
            : [],

          modelMetrics: Array.isArray(pipeline.modelMetrics)
            ? pipeline.modelMetrics
            : [],

          drift: Array.isArray(pipeline.drift) ? pipeline.drift : [],
          forecast: Array.isArray(pipeline.forecast) ? pipeline.forecast : [],
          detailedForecast: Array.isArray(pipeline.detailedForecast)
            ? pipeline.detailedForecast
            : [],

          alerts: Array.isArray(pipeline.alerts) ? pipeline.alerts : [],

          device: pipeline.device || {},
          sensor: pipeline.sensor || {},
          regime: pipeline.regime || {
            current: "Unknown",
            confidence: 0,
          },
          reliability: pipeline.reliability || {
            trust: 0,
            uptime: 0,
            validity: 0,
            driftSigma: 0,
          },
          anomaly: pipeline.anomaly || {},

          // ✅ FIXED mapping
          particleSize: normalize(physics.particleSize),
          densityScatter: normalize(physics.densityScatter),
          clusters: normalize(physics.clusters),
          transitions: transitions,
        };

        console.log("✅ FINAL SNAPSHOT:", safeData); // DEBUG

        setSnapshot(safeData);
        setError(null);
      } catch (err) {
        console.error("API ERROR:", err);
        setError("Failed to load data");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <ResilienceContext.Provider value={{ snapshot, loading, error }}>
      {children}
    </ResilienceContext.Provider>
  );
}

export function useSensorData() {
  return useContext(ResilienceContext);
}

export function formatRelative(date: string) {
  const diff = (Date.now() - new Date(date).getTime()) / 1000;

  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}