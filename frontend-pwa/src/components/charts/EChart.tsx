import { useEffect, useState } from "react";
import type { ComponentType } from "react";
import { ChartSkeleton } from "./ChartSkeleton";

type EChartsProps = {
  option: Record<string, unknown>;
  style?: React.CSSProperties;
  notMerge?: boolean;
  lazyUpdate?: boolean;
  className?: string;
};

/**
 * Client-only ECharts wrapper. echarts-for-react does not render correctly
 * during SSR (returns a non-component object), so we lazy-load it after
 * hydration to avoid "Element type is invalid" errors.
 */
export function EChart({ style, ...props }: EChartsProps) {
  const [Comp, setComp] = useState<ComponentType<EChartsProps> | null>(null);

  useEffect(() => {
    let mounted = true;
    import("echarts-for-react").then((mod) => {
      if (mounted) setComp(() => mod.default as unknown as ComponentType<EChartsProps>);
    });
    return () => {
      mounted = false;
    };
  }, []);

  if (!Comp) {
    const h = typeof style?.height === "number" ? style.height : 240;
    return <ChartSkeleton height={h} />;
  }
  return <Comp {...props} style={style} />;
}
