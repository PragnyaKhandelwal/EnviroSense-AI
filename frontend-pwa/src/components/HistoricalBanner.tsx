import { CloudOff } from "lucide-react";
import { useSensorData, formatRelative } from "@/lib/resilience";

/**
 * Subtle advisory banner shown on AI / Prediction / Alerts panels when the
 * backend is unreachable. Communicates that the data on screen is cached.
 */
export function HistoricalBanner({ message }: { message?: string }) {
  const { isLive, lastSyncTs } = useSensorData();
  if (isLive) return null;
  return (
    <div
      className="mb-3 flex items-center gap-2 rounded-md border border-moderate/30 bg-moderate/10 px-3 py-2 text-[11px] text-moderate"
      role="status"
    >
      <CloudOff className="h-3.5 w-3.5 shrink-0" />
      <span className="flex-1 leading-snug">
        {message ?? "Viewing cached AI forecast. Live inference paused."}
      </span>
      <span className="hidden sm:inline text-muted-foreground font-mono">
        last sync {formatRelative(lastSyncTs)}
      </span>
    </div>
  );
}
