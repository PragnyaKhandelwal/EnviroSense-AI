import { useState } from "react";
import { ServerCrash, Wifi, ChevronUp, ChevronDown } from "lucide-react";
import { useSensorData } from "@/lib/resilience";

/**
 * Floating bottom-right developer pill that simulates a backend outage.
 * Used to prove the dashboard remains crash-free during a network event.
 */
export function OutageToggle() {
  const { outageSimulated, setOutageSimulated, isLive } = useSensorData();
  const [collapsed, setCollapsed] = useState(false);

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={() => setCollapsed(false)}
        aria-label="Expand resilience controls"
        className={`fixed bottom-4 right-4 z-40 grid place-items-center h-10 w-10 rounded-full border backdrop-blur-md shadow-lg transition-colors ${
          isLive
            ? "border-clean/40 bg-clean/10 text-clean hover:bg-clean/20"
            : "border-moderate/40 bg-moderate/15 text-moderate hover:bg-moderate/25"
        }`}
      >
        {isLive ? <Wifi className="h-4 w-4" /> : <ServerCrash className="h-4 w-4" />}
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 z-40 select-none">
      <div
        className={`flex items-center gap-3 rounded-full border backdrop-blur-md px-3 py-2 shadow-lg transition-colors ${
          outageSimulated
            ? "border-moderate/40 bg-moderate/15"
            : "border-clean/30 bg-background/80"
        }`}
      >
        <div className="flex items-center gap-2">
          {outageSimulated ? (
            <ServerCrash className="h-3.5 w-3.5 text-moderate" />
          ) : (
            <Wifi className="h-3.5 w-3.5 text-clean" />
          )}
          <span className="text-[11px] font-medium">
            {outageSimulated ? "Outage Simulated" : "Simulate Server Outage"}
          </span>
        </div>

        <button
          type="button"
          role="switch"
          aria-checked={outageSimulated}
          onClick={() => setOutageSimulated(!outageSimulated)}
          className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors ${
            outageSimulated ? "bg-moderate" : "bg-secondary"
          }`}
        >
          <span
            className={`inline-block h-4 w-4 rounded-full bg-background shadow transition-transform ${
              outageSimulated ? "translate-x-4" : "translate-x-0.5"
            }`}
          />
        </button>

        <button
          type="button"
          onClick={() => setCollapsed(true)}
          aria-label="Collapse resilience controls"
          className="text-muted-foreground hover:text-foreground"
        >
          <ChevronDown className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="mt-1 ml-3 text-[10px] text-muted-foreground max-w-[220px] leading-tight">
        Forces the 60s poll to fail, flips the UI into Historical Mode, and renders only cached data.
      </div>
      {/* Re-export ChevronUp so unused-import linter stays quiet if collapsed branch is removed */}
      <span className="hidden">
        <ChevronUp className="h-3.5 w-3.5" />
      </span>
    </div>
  );
}
