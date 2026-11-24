import { createFileRoute, useRouter } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Download } from "lucide-react";
import { getDocument } from "@/lib/api";
import { CategoryBadge, ErrorState } from "@/components/search/Results";
import { Skeleton } from "@/components/ui/skeleton";

export const Route = createFileRoute("/doc/$id")({
  validateSearch: (s: Record<string, unknown>): { s?: number } => {
    const n = Number(s.s);
    return { s: s.s !== undefined && Number.isFinite(n) ? n : undefined };
  },
  head: ({ params }) => ({ meta: [{ title: `Document ${params.id} - SemanticCache` }] }),
  component: DocumentPage,
});

function DocumentPage() {
  const { id } = Route.useParams();
  const { s: similarity } = Route.useSearch();
  const router = useRouter();
  const docId = Number(id);

  const { data, isPending, error, refetch } = useQuery({
    queryKey: ["doc", docId],
    queryFn: ({ signal }) => getDocument(docId, signal),
    enabled: Number.isInteger(docId),
    retry: false,
    staleTime: Infinity,
  });

  function back() {
    // Return to the previous results page if there is one; otherwise go to the explorer.
    if (window.history.length > 1) router.history.back();
    else router.navigate({ to: "/" });
  }

  function exportJson() {
    if (!data) return;
    const payload = {
      ...data,
      similarity_score: similarity ?? null,
      exported_at: new Date().toISOString(),
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `document_${data.doc_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const paragraphs = data?.clean_text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);

  return (
    <div className="animate-fade-in">
      <nav
        aria-label="Breadcrumb"
        className="mb-5 flex items-center gap-2 text-sm text-muted-foreground"
      >
        <button
          type="button"
          onClick={back}
          className="inline-flex items-center gap-1.5 transition-colors hover:text-brand"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden /> Results
        </button>
        <span aria-hidden>/</span>
        <span className="text-foreground">Document {id}</span>
      </nav>

      {error ? (
        <ErrorState message={(error as Error).message} onRetry={() => refetch()} />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
          <article className="panel min-w-0 p-6 sm:p-8">
            {data ? (
              <>
                <CategoryBadge category={data.category} />
                <h1 className="mt-4 text-[28px] font-semibold tracking-tight">
                  Document {data.doc_id}
                </h1>
                <div className="my-5 h-px bg-border" />
                <div className="max-w-[68ch] space-y-4 text-[15px] leading-[1.75] text-foreground/85">
                  {paragraphs && paragraphs.length > 0 ? (
                    paragraphs.map((p, i) => (
                      <p key={i} className="whitespace-pre-wrap break-words">
                        {p}
                      </p>
                    ))
                  ) : (
                    <p className="text-muted-foreground">This document has no text content.</p>
                  )}
                </div>
              </>
            ) : (
              <div className="space-y-3" aria-busy={isPending}>
                <Skeleton className="h-6 w-28" />
                <Skeleton className="h-8 w-48" />
                {Array.from({ length: 8 }).map((_, i) => (
                  <Skeleton key={i} className="h-4" style={{ width: `${70 + (i % 4) * 8}%` }} />
                ))}
              </div>
            )}
          </article>

          <aside className="space-y-3 lg:sticky lg:top-20 lg:self-start">
            <div className="panel p-4">
              <div className="label-caps">Similarity to query</div>
              {similarity !== undefined ? (
                <>
                  <div className="mt-2 font-mono text-2xl font-semibold text-brand">
                    {similarity.toFixed(2)}
                  </div>
                  <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-border">
                    <div
                      className="h-full rounded-full bg-brand"
                      style={{ width: `${Math.min(100, Math.max(0, similarity * 100))}%` }}
                    />
                  </div>
                </>
              ) : (
                <div className="mt-2 text-sm text-muted-foreground">Not available</div>
              )}
            </div>
            <div className="panel p-4">
              <div className="label-caps">Category</div>
              <div className="mt-2 font-mono text-sm">{data?.category ?? "..."}</div>
            </div>
            <div className="panel p-4">
              <div className="label-caps">Dominant cluster</div>
              <div className="mt-2 font-mono text-sm">
                {data ? `Cluster ${data.dominant_cluster}` : "..."}
              </div>
              <div className="mt-1 text-xs text-muted-foreground">GMM soft assignment</div>
            </div>
            <button
              type="button"
              onClick={exportJson}
              disabled={!data}
              className="flex h-10 w-full items-center justify-center gap-2 rounded-md border border-border text-sm font-medium transition-colors hover:border-brand/50 hover:text-brand disabled:opacity-50"
            >
              <Download className="h-4 w-4" aria-hidden /> Export JSON
            </button>
          </aside>
        </div>
      )}
    </div>
  );
}
