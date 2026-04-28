import type { ReactNode } from "react";
import { ErrorBoundary, ChartFallback } from "./ErrorBoundary";

/**
 * Wrap any chart in this to make it crash-proof. If the inner chart throws
 * (bad data, undefined arrays, ECharts internal errors), we render a sleek
 * "Data visualization temporarily unavailable" panel instead of a white screen.
 */
export function SafeChart({
  children,
  label,
  height = 240,
  resetKey,
}: {
  children: ReactNode;
  label?: string;
  height?: number;
  resetKey?: string | number;
}) {
  return (
    <ErrorBoundary
      label={label}
      resetKey={resetKey}
      fallback={<ChartFallback height={height} />}
    >
      {children}
    </ErrorBoundary>
  );
}
