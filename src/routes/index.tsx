import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";

import { Layout } from "@/components/Layout";
import { PageHeader } from "@/components/PageHeader";
import { LiveStatusCard } from "@/components/cards/LiveStatusCard";
import { ReliabilityCard } from "@/components/cards/ReliabilityCard";
import { RegimeCard } from "@/components/cards/RegimeCard";
import { AnomalyStatusCard } from "@/components/cards/AnomalyStatusCard";
import { Timeline24hChart } from "@/components/charts/Timeline24hChart";
import { SafeChart } from "@/components/SafeChart";
import { HistoricalBanner } from "@/components/HistoricalBanner";
import { GenerateReportButton, ReportDrawer } from "@/components/ReportDrawer";

export const Route = createFileRoute("/")({
  component: OverviewPage,
});

function OverviewPage() {
  const navigate = useNavigate();

  const [reportOpen, setReportOpen] = useState(false);
  const [loading, setLoading] = useState(true);

   const [timelineData, setTimelineData] = useState<any[]>([]);

  useEffect(() => {
    const init = async () => {
      try {
        
        const res = await fetch("/api/me", {
          credentials: "include",
        });

        if (!res.ok) {
          navigate({ to: "/login" });
          return;
        }

        const apiRes = await fetch("/api/forecast")
        const apiData = await apiRes.json();

        console.log("API DATA:", apiData); // debug

        setTimelineData(apiData); 
        setLoading(false);
      } catch (err) {
        console.error(err);
        navigate({ to: "/login" });
      }
    };

    init();
  }, []);

  if (loading) return <div>Loading...</div>;

  const description =
    "Real-time air-quality readings and AI pipeline outputs";

  return (
    <Layout>
      <PageHeader
        eyebrow="Operational Layer"
        title="Overview"
        description={description}
        actions={
          <GenerateReportButton onClick={() => setReportOpen(true)} />
        }
      />

      <HistoricalBanner />
      <ReportDrawer
        open={reportOpen}
        onClose={() => setReportOpen(false)}
      />

      {/* Hero row */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <div className="lg:col-span-2">
          <LiveStatusCard />
        </div>

        {/* temporarily static (you can connect later) */}
        <ReliabilityCard />
        <RegimeCard />
      </div>

      {/* Timeline */}
      <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <section className="panel p-5 lg:col-span-2">
          <div className="flex items-end justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted">
                24-Hour Timeline
              </div>
              <div className="mt-0.5 text-sm">
                PM2.5 overlaid with temperature & humidity
              </div>
            </div>
          </div>

          <div className="mt-3">
            <SafeChart label="timeline-24h" height={320}>
              <Timeline24hChart data={timelineData} />
            </SafeChart>
          </div>
        </section>

        <AnomalyStatusCard />
      </div>
    </Layout>
  );
}
