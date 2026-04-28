import { ShieldCheck, Users } from "lucide-react";

export function Footer() {
  return (
    <footer className="mt-10 panel p-5 text-sm text-muted-foreground">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        
        <div>
          <div className="font-medium text-foreground">
          System Overview
          </div>

          <p className="mt-1 max-w-2xl text-xs leading-relaxed">
          EnviroSense AI acts as an autonomous ecological sentinel, providing high-fidelity environmental monitoring through a dedicated localized node. By leveraging a high-frequency sampling architecture, the platform translates raw environmental variables into actionable regional insights. This interface demonstrates an end-to-end engineering pipeline designed for real-time climate modeling and atmospheric data integrity.
          </p>
        </div>

      </div>
    </footer>
  );
}
