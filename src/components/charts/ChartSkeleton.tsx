export function ChartSkeleton({ height = 240 }: { height?: number }) {
  return (
    <div
      className="relative w-full overflow-hidden rounded-md border border-border/60 bg-panel/40"
      style={{ height }}
    >
      <div className="absolute inset-0 grid-bg opacity-[0.08]" />
      <div className="absolute inset-0 flex items-end gap-1 px-3 pb-3">
        {Array.from({ length: 24 }).map((_, i) => (
          <div
            key={i}
            className="flex-1 rounded-sm bg-muted/40 animate-pulse"
            style={{ height: `${20 + ((i * 13) % 60)}%`, animationDelay: `${i * 40}ms` }}
          />
        ))}
      </div>
    </div>
  );
}
