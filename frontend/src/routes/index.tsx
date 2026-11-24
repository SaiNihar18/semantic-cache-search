import { useEffect, useState } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { searchDocs } from "@/lib/api";
import { recordOp } from "@/lib/ops";
import { SearchConsole, THRESHOLD_MAX, THRESHOLD_MIN } from "@/components/search/SearchConsole";
import {
  EmptyState,
  ErrorState,
  ResultCard,
  ResultsSkeleton,
  SummaryBar,
} from "@/components/search/Results";

type Search = { q?: string; t?: number };

export const Route = createFileRoute("/")({
  validateSearch: (s: Record<string, unknown>): Search => {
    const t = Number(s.t);
    return {
      q: typeof s.q === "string" && s.q.trim() ? s.q : undefined,
      t: Number.isFinite(t) ? Math.min(THRESHOLD_MAX, Math.max(THRESHOLD_MIN, t)) : undefined,
    };
  },
  head: () => ({
    meta: [
      { title: "SemanticCache - Intelligent Document Search" },
      {
        name: "description",
        content:
          "Fast, cluster-aware document retrieval with intelligent caching across 20,000 documents.",
      },
    ],
  }),
  component: Explorer,
});

function Explorer() {
  const { q = "", t = 0.85 } = Route.useSearch();
  const navigate = useNavigate({ from: "/" });

  const { data, error, isFetching, refetch } = useQuery({
    queryKey: ["search", q, t],
    enabled: q.length > 0,
    // A given (query, threshold) result is reused when returning from a document.
    staleTime: Infinity,
    retry: false,
    queryFn: async ({ signal }) => {
      const res = await searchDocs(q, t, signal);
      recordOp({
        query: q.length > 40 ? `${q.slice(0, 40)}…` : q,
        action: res.cache_hit ? "HIT" : "MISS",
        durationMs: res.elapsedMs,
      });
      return res;
    },
  });

  // After a few seconds without a response, explain that the free-tier server may be waking up.
  const [waking, setWaking] = useState(false);
  useEffect(() => {
    if (!isFetching) return setWaking(false);
    const id = setTimeout(() => setWaking(true), 6000);
    return () => clearTimeout(id);
  }, [isFetching]);

  function onSubmit(query: string, threshold: number) {
    if (query === q && threshold === t) refetch();
    else navigate({ search: { q: query, t: threshold } });
  }

  return (
    <div className="space-y-8">
      <header className="mx-auto max-w-[680px] text-center">
        <h1 className="text-[32px] font-semibold leading-tight tracking-tight sm:text-[40px]">
          Semantic Cache Search
        </h1>
        <p className="mt-3 text-base text-muted-foreground">
          Cluster-aware retrieval over 20,000 newsgroup posts. Rephrased questions are answered from
          cache.
        </p>
      </header>

      <SearchConsole query={q} threshold={t} loading={isFetching} onSubmit={onSubmit} />

      <section aria-label="Results" aria-live="polite" className="min-h-[240px]">
        {isFetching ? (
          <ResultsSkeleton waking={waking} />
        ) : error ? (
          <ErrorState message={(error as Error).message} onRetry={() => refetch()} />
        ) : data ? (
          <div className="animate-fade-in space-y-3">
            <SummaryBar data={data} />
            {data.results.map((r, i) => (
              <ResultCard key={`${r.doc_id}-${i}`} item={r} rank={i + 1} />
            ))}
          </div>
        ) : (
          <EmptyState />
        )}
      </section>
    </div>
  );
}
