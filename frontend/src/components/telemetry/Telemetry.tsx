import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Search, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { clearCacheApi } from "@/lib/api";
import { useStats } from "@/lib/hooks";
import { clearOps, useOps } from "@/lib/ops";
import { ErrorState } from "@/components/search/Results";
import { Skeleton } from "@/components/ui/skeleton";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

function StatCard({
  label,
  children,
  hint,
}: {
  label: string;
  children: React.ReactNode;
  hint: string;
}) {
  return (
    <div className="panel p-4">
      <div className="label-caps">{label}</div>
      <div className="mt-2 text-3xl font-semibold tracking-tight">{children}</div>
      <div className="mt-2 text-xs text-muted-foreground">{hint}</div>
    </div>
  );
}

function Ring({ pct }: { pct: number }) {
  const R = 20;
  const C = 2 * Math.PI * R;
  return (
    <svg width="48" height="48" viewBox="0 0 48 48" className="-rotate-90" aria-hidden>
      <circle cx="24" cy="24" r={R} stroke="var(--border)" strokeWidth="4" fill="none" />
      <circle
        cx="24"
        cy="24"
        r={R}
        stroke="var(--brand)"
        strokeWidth="4"
        fill="none"
        strokeLinecap="round"
        strokeDasharray={C}
        strokeDashoffset={C - (pct / 100) * C}
        style={{ transition: "stroke-dashoffset 0.4s ease" }}
      />
    </svg>
  );
}

export function Telemetry() {
  const qc = useQueryClient();
  const { data: stats, isPending, isError, error, refetch, isFetching } = useStats();
  const ops = useOps();
  const [confirmOpen, setConfirmOpen] = useState(false);

  async function clearCache() {
    try {
      await clearCacheApi();
      clearOps();
      qc.removeQueries({ queryKey: ["search"] });
      await qc.invalidateQueries({ queryKey: ["stats"] });
      toast.success("Cache cleared");
    } catch {
      toast.error("Failed to clear the cache");
    }
  }

  const hits = stats?.hit_count ?? 0;
  const misses = stats?.miss_count ?? 0;
  const total = hits + misses;
  const hitRate = (stats?.hit_rate ?? 0) * 100;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-[28px] font-semibold tracking-tight">Metrics</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Server-wide cache performance across all visitors. Refreshes every 10 seconds.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => refetch()}
            aria-label="Refresh stats"
            className="grid h-9 w-9 place-items-center rounded-md border border-border text-muted-foreground transition-colors hover:text-foreground"
          >
            <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
          </button>
          <button
            type="button"
            onClick={() => setConfirmOpen(true)}
            className="inline-flex h-9 items-center gap-2 rounded-md border border-destructive/40 px-3 text-sm font-medium text-destructive transition-colors hover:bg-destructive/10"
          >
            <Trash2 className="h-4 w-4" aria-hidden /> Clear cache
          </button>
        </div>
      </div>

      {isError ? (
        <ErrorState message={(error as Error).message} onRetry={() => refetch()} />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {isPending ? (
            [0, 1, 2].map((i) => <Skeleton key={i} className="h-[116px] rounded-xl" />)
          ) : (
            <>
              <StatCard label="Hit rate" hint={`${total} total queries`}>
                <div className="flex items-center justify-between">
                  <span>
                    {Math.round(hitRate)}
                    <span className="text-lg text-muted-foreground">%</span>
                  </span>
                  <Ring pct={hitRate} />
                </div>
              </StatCard>
              <StatCard label="Cache entries" hint="Distinct queries stored">
                {stats.total_entries}
              </StatCard>
              <StatCard label="Hits / misses" hint={total ? "Share of hits" : "No queries yet"}>
                <span className="text-brand">{hits}</span>
                <span className="px-1.5 text-lg text-muted-foreground">/</span>
                <span className="text-destructive">{misses}</span>
                <div
                  className="mt-3 h-1.5 overflow-hidden rounded-full bg-border"
                  role="img"
                  aria-label={`${hits} hits and ${misses} misses`}
                >
                  <div
                    className="h-full bg-brand transition-all duration-500"
                    style={{ width: total ? `${(hits / total) * 100}%` : "0%" }}
                  />
                </div>
              </StatCard>
            </>
          )}
        </div>
      )}

      <section className="panel p-5" aria-labelledby="ops-title">
        <div className="mb-4 flex items-center justify-between">
          <h2 id="ops-title" className="text-lg font-semibold">
            Recent operations
          </h2>
          <span className="text-xs text-muted-foreground">Your searches in this browser only</span>
        </div>
        {ops.length === 0 ? (
          <div className="flex flex-col items-center gap-2 py-10 text-center text-muted-foreground">
            <Search className="h-5 w-5 opacity-50" aria-hidden />
            <p className="text-sm">Run a search to see cache operations here.</p>
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Time</TableHead>
                <TableHead>Query</TableHead>
                <TableHead>Result</TableHead>
                <TableHead className="text-right">Duration</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody className="font-mono text-xs">
              {ops.map((op) => (
                <TableRow key={op.id}>
                  <TableCell className="text-muted-foreground">
                    {op.time.toLocaleTimeString()}
                  </TableCell>
                  <TableCell className="max-w-[240px] truncate" title={op.query}>
                    {op.query}
                  </TableCell>
                  <TableCell>
                    <span
                      className={`rounded px-1.5 py-0.5 font-bold ${
                        op.action === "HIT"
                          ? "bg-brand/15 text-brand"
                          : "bg-warning/15 text-warning"
                      }`}
                    >
                      {op.action}
                    </span>
                  </TableCell>
                  <TableCell className="text-right text-muted-foreground">
                    {op.durationMs} ms
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </section>

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear the cache?</AlertDialogTitle>
            <AlertDialogDescription>
              This removes every cached query and resets hit/miss counters for all visitors. This
              cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={clearCache}>Clear cache</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
