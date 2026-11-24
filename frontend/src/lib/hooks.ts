import { useQuery } from "@tanstack/react-query";
import { getStats } from "./api";

/** Server-wide cache stats, polled so the Metrics page and status pill stay fresh. */
export function useStats(intervalMs = 10_000) {
  return useQuery({
    queryKey: ["stats"],
    queryFn: ({ signal }) => getStats(signal),
    refetchInterval: intervalMs,
    retry: 1,
  });
}

export type BackendStatus = "online" | "offline" | "connecting";

export function useBackendStatus(): BackendStatus {
  const { data, isError, isPending } = useStats();
  if (data) return "online";
  if (isError) return "offline";
  return isPending ? "connecting" : "offline";
}
