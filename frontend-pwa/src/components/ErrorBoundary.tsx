import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  /** Optional label so logs can identify which boundary tripped. */
  label?: string;
  /** Reset key — when this changes, the boundary clears its error. */
  resetKey?: string | number;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Catches render errors anywhere in its subtree. Used to wrap individual charts
 * and AI cards so a single malformed payload can't take the whole dashboard down.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Log but never re-throw — graceful degradation is the contract.
    // eslint-disable-next-line no-console
    console.warn(`[ErrorBoundary:${this.props.label ?? "anon"}]`, error.message, info.componentStack);
  }

  componentDidUpdate(prev: Props) {
    if (this.state.hasError && prev.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false, error: null });
    }
  }

  reset = () => this.setState({ hasError: false, error: null });

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? <ChartFallback onRetry={this.reset} />;
    }
    return this.props.children;
  }
}

/** Default fallback — sleek, on-brand, used by SafeChart when nothing custom is provided. */
export function ChartFallback({
  onRetry,
  height = 240,
  message = "Data visualization temporarily unavailable",
}: {
  onRetry?: () => void;
  height?: number;
  message?: string;
}) {
  return (
    <div
      className="relative w-full overflow-hidden rounded-md border border-moderate/30 bg-moderate/5"
      style={{ height }}
      role="alert"
    >
      <div className="absolute inset-0 grid-bg opacity-[0.06] pointer-events-none" />
      <div className="relative h-full flex flex-col items-center justify-center gap-2 text-center px-4">
        <AlertTriangle className="h-5 w-5 text-moderate" />
        <div className="text-xs font-medium text-moderate">{message}</div>
        <div className="text-[10px] text-muted-foreground max-w-[28ch]">
          The chart received malformed or partial data. Cached values remain visible across the dashboard.
        </div>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-1 inline-flex items-center gap-1.5 rounded-md border border-border bg-panel px-2.5 py-1 text-[10px] text-muted-foreground hover:text-foreground hover:border-clean/40 transition-colors"
          >
            <RefreshCw className="h-3 w-3" />
            Retry render
          </button>
        )}
      </div>
    </div>
  );
}
