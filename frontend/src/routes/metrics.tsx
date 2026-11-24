import { createFileRoute } from "@tanstack/react-router";
import { Telemetry } from "@/components/telemetry/Telemetry";

export const Route = createFileRoute("/metrics")({
  head: () => ({ meta: [{ title: "Metrics - SemanticCache" }] }),
  component: Telemetry,
});
