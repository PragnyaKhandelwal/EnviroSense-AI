import { ShieldCheck, Users } from "lucide-react";

export function Footer() {
  return (
    <footer className="mt-10 panel p-5 text-sm text-muted-foreground">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        
        <div>
          <div className="font-medium text-foreground">
            Case study · Personal pollutant exposure
          </div>

          <p className="mt-1 max-w-2xl text-xs leading-relaxed">
            EnviroSense AI is a hyper-local ecological node combining domestic sensor data
            (~70k 1-min samples from "My Terrace-on-Room") with regional context. All metrics
            shown here are illustrative of the engineering pipeline and modeling approach.
          </p>
        </div>

      </div>
    </footer>
  );
}