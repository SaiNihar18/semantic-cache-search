import { Link } from "@tanstack/react-router";
import { AlertCircle, ArrowRight, RefreshCw, Search, ServerCrash } from "lucide-react";
import { deriveTitle, type ResultItem, type SearchResult } from "@/lib/api";
import { GROUP_META, groupOf } from "@/lib/categories";
import { Skeleton } from "@/components/ui/skeleton";

export function CategoryBadge({ category }: { category: string }) {
  const meta = GROUP_META[groupOf(category)];
  return (
    <span className="inline-flex items-center gap-2 rounded-md border border-border bg-secondary/60 px-2 py-0.5 font-mono text-xs text-foreground/85">
      <span className={`h-1.5 w-1.5 rounded-full ${meta.dot}`} aria-hidden />
      {category}
    </span>
  );
}

export function SummaryBar({ data }: { data: SearchResult }) {
  const hit = data.cache_hit;
  return (
    <div
      className={`panel flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 ${
        hit ? "border-brand/40" : "border-warning/40"
      }`}
    >
      <span
        className={`inline-flex items-center gap-2 rounded-md px-2 py-0.5 font-mono text-xs font-bold ${
          hit ? "bg-brand/15 text-brand" : "bg-warning/15 text-warning"
        }`}
      >
        <span className={`h-1.5 w-1.5 rounded-full ${hit ? "bg-brand" : "bg-warning"}`} />
        CACHE {hit ? "HIT" : "MISS"}
      </span>
      <div className="min-w-0 flex-1 text-sm text-muted-foreground">
        {hit ? (
          <span className="line-clamp-2">
            Matched <span className="text-foreground">&ldquo;{data.matched_query}&rdquo;</span> at{" "}
            <span className="font-mono text-brand">{(data.similarity_score ?? 0).toFixed(2)}</span>{" "}
            similarity
          </span>
        ) : (
          <span>New query: searched all documents and stored the result</span>
        )}
      </div>
      <div className="flex items-center gap-4 font-mono text-xs text-muted-foreground">
        <span>{data.elapsedMs} ms</span>
        <span>Cluster {data.dominant_cluster}</span>
        <span>{data.results.length} results</span>
      </div>
    </div>
  );
}

export function ResultCard({ item, rank }: { item: ResultItem; rank: number }) {
  const pct = Math.max(0, Math.min(100, item.similarity_score * 100));
  return (
    <Link
      to="/doc/$id"
      params={{ id: String(item.doc_id) }}
      search={{ s: Number(item.similarity_score.toFixed(4)) }}
      className="panel group block p-5 transition-colors hover:border-brand/40 hover:bg-secondary/40"
    >
      <div className="flex items-start justify-between gap-4">
        <CategoryBadge category={item.category} />
        <div className="flex shrink-0 items-center gap-2" title="Cosine similarity to your query">
          <div className="h-1.5 w-16 overflow-hidden rounded-full bg-border" aria-hidden>
            <div className="h-full rounded-full bg-brand" style={{ width: `${pct}%` }} />
          </div>
          <span className="font-mono text-xs text-foreground">
            {item.similarity_score.toFixed(2)}
          </span>
        </div>
      </div>
      <h3 className="mt-3 text-base font-semibold leading-snug tracking-tight">
        <span className="mr-2 text-muted-foreground">{rank}.</span>
        {deriveTitle(item.text)}
      </h3>
      <p className="mt-2 line-clamp-3 text-sm leading-relaxed text-muted-foreground">{item.text}</p>
      <div className="mt-4 flex items-center justify-between border-t border-border pt-3 text-xs">
        <span className="font-mono text-muted-foreground">
          Doc {item.doc_id} · Cluster {item.dominant_cluster}
        </span>
        <span className="inline-flex items-center gap-1 font-medium text-brand">
          Read article
          <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
        </span>
      </div>
    </Link>
  );
}

export function ResultsSkeleton({ waking }: { waking: boolean }) {
  return (
    <div className="space-y-3" aria-busy="true" aria-live="polite">
      {waking && (
        <div className="panel flex items-center gap-3 border-warning/40 px-4 py-3 text-sm text-warning">
          <RefreshCw className="h-4 w-4 animate-spin" aria-hidden />
          Waking up the server. The free tier can take up to a minute on the first request.
        </div>
      )}
      <Skeleton className="h-12 w-full rounded-xl" />
      {[0, 1, 2].map((i) => (
        <div key={i} className="panel space-y-3 p-5">
          <div className="flex justify-between">
            <Skeleton className="h-5 w-28" />
            <Skeleton className="h-4 w-20" />
          </div>
          <Skeleton className="h-5 w-3/5" />
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-5/6" />
        </div>
      ))}
    </div>
  );
}

export function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-3 py-16 text-center">
      <div className="grid h-12 w-12 place-items-center rounded-full border border-border bg-card text-muted-foreground">
        <Search className="h-5 w-5" aria-hidden />
      </div>
      <p className="text-sm text-foreground">Search the 20 Newsgroups corpus by meaning</p>
      <p className="max-w-sm text-xs text-muted-foreground">
        Type a question or pick an example above. Results from similar earlier queries are served
        straight from the semantic cache.
      </p>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="panel flex items-start gap-3 border-destructive/40 p-4">
      <ServerCrash className="mt-0.5 h-5 w-5 shrink-0 text-destructive" aria-hidden />
      <div className="flex-1">
        <p className="text-sm font-medium">Something went wrong</p>
        <p className="mt-1 flex items-start gap-1.5 text-sm text-muted-foreground">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          {message}
        </p>
      </div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="rounded-md border border-border px-3 py-1.5 text-sm transition-colors hover:bg-accent"
        >
          Retry
        </button>
      )}
    </div>
  );
}
