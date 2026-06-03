import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  ArrowRight,
  SlidersHorizontal,
  Database,
  Folder,
  ArrowLeft,
  Trash2,
  BarChart3,
  ListChecks,
  ArrowLeftRight,
  Timer,
  Sparkles,
  RefreshCw,
  Download,
} from "lucide-react";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";

const API = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";

const SUGGESTED = [
  "Space & Rockets",
  "Computer Hardware",
  "Religion & Atheism",
  "Sports",
  "Medicine",
];

type ResultItem = {
  doc_id: number;
  category: string;
  text: string;
  similarity_score: number;
  dominant_cluster: number;
};

type QueryResponse = {
  query: string;
  cache_hit: boolean;
  matched_query: string | null;
  similarity_score: number | null;
  result: string;
  results: ResultItem[];
  dominant_cluster: number;
};

type CacheStats = {
  total_entries: number;
  hit_count: number;
  miss_count: number;
  hit_rate: number;
};

type FullDoc = {
  doc_id: number;
  category: string;
  clean_text: string;
  dominant_cluster: number;
};

type SavedState = {
  query: string;
  threshold: number;
  response: QueryResponse;
  scrollY: number;
  responseTimeMs: number | null;
};

function categoryTint(category: string) {
  if (category.startsWith("sci."))
    return "bg-[rgba(62,207,142,0.12)] text-[#3ECF8E] border-[rgba(62,207,142,0.3)]";
  if (category.startsWith("comp."))
    return "bg-[rgba(168,120,255,0.12)] text-[#B795FF] border-[rgba(168,120,255,0.3)]";
  if (category.startsWith("rec."))
    return "bg-[rgba(244,189,80,0.12)] text-[#F4BD50] border-[rgba(244,189,80,0.3)]";
  return "bg-[rgba(200,200,200,0.08)] text-[#A8A8A8] border-[rgba(200,200,200,0.2)]";
}

export const Route = createFileRoute("/")(
  {
    head: () => ({
      meta: [
        { title: "SemanticCache — Intelligent Document Search" },
        {
          name: "description",
          content:
            "Fast, cluster-aware document retrieval with intelligent caching across 20,000 documents.",
        },
      ],
    }),
    component: Index,
  },
);

function Index() {
  const [query, setQuery] = useState("");
  const [threshold, setThreshold] = useState(0.85);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [responseTimeMs, setResponseTimeMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<CacheStats | null>(null);
  const [inputError, setInputError] = useState(false);

  const [article, setArticle] = useState<FullDoc | null>(null);
  const [articleLoading, setArticleLoading] = useState(false);
  const [articleSimilarity, setArticleSimilarity] = useState<number | null>(null);
  const savedRef = useRef<SavedState | null>(null);

  // Active nav tab for top navbar
  const [activeNav, setActiveNav] = useState("Explorer");

  useEffect(() => {
    refreshStats();
  }, []);

  async function refreshStats() {
    try {
      const r = await fetch(`${API}/cache/stats`);
      if (r.ok) setStats(await r.json());
    } catch {
      /* ignore */
    }
  }

  async function runSearch(q: string) {
    const trimmed = q.trim();
    if (!trimmed) {
      setInputError(true);
      setTimeout(() => setInputError(false), 600);
      return;
    }
    setError(null);
    setLoading(true);
    setResponse(null);
    const start = performance.now();
    try {
      const r = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmed, similarity_threshold: threshold }),
      });
      if (!r.ok) throw new Error("bad");
      const data: QueryResponse = await r.json();
      setResponse(data);
      setResponseTimeMs(Math.round(performance.now() - start));
      refreshStats();
    } catch {
      setError(
        "Failed to reach the search API. Make sure the backend is running on localhost:8000.",
      );
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit() {
    runSearch(query);
  }

  function onPill(p: string) {
    setQuery(p);
    runSearch(p);
  }

  function onKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") handleSubmit();
  }

  async function openArticle(item: ResultItem) {
    if (!response) return;
    savedRef.current = {
      query,
      threshold,
      response,
      scrollY: window.scrollY,
      responseTimeMs,
    };
    setArticleSimilarity(item.similarity_score);
    setArticleLoading(true);
    setArticle({
      doc_id: item.doc_id,
      category: item.category,
      clean_text: "",
      dominant_cluster: item.dominant_cluster,
    });
    try {
      const r = await fetch(`${API}/document/${item.doc_id}`);
      if (!r.ok) throw new Error();
      const data: FullDoc = await r.json();
      setArticle(data);
    } catch {
      setError("Failed to load the document.");
      setArticle(null);
    } finally {
      setArticleLoading(false);
    }
  }

  function backToResults() {
    setArticle(null);
    setArticleSimilarity(null);
    if (savedRef.current) {
      const s = savedRef.current;
      setQuery(s.query);
      setThreshold(s.threshold);
      setResponse(s.response);
      setResponseTimeMs(s.responseTimeMs);
      requestAnimationFrame(() => window.scrollTo({ top: s.scrollY }));
    }
  }

  async function clearCache() {
    try {
      const r = await fetch(`${API}/cache`, { method: "DELETE" });
      if (!r.ok) throw new Error();
      toast.success("Cache cleared");
      refreshStats();
    } catch {
      toast.error("Failed to clear cache");
    }
  }

  const sliderPct = ((threshold - 0.7) / (0.98 - 0.7)) * 100;

  return (
    <div className="min-h-screen w-full text-foreground" style={{ background: "#0d1117" }}>
      {/* ambient bg glow */}
      <div
        className="pointer-events-none fixed inset-0 -z-10"
        style={{
          backgroundImage:
            "radial-gradient(800px 400px at 50% -100px, rgba(62,207,142,0.07), transparent 60%), radial-gradient(600px 300px at 90% 80%, rgba(62,207,142,0.04), transparent 70%)",
        }}
      />
      <Toaster theme="dark" />

      {/* ═══════ TOP NAVBAR (matches reference card: image.png / image (5).png) ═══════ */}
      <nav className="border-b border-white/[0.06] bg-[rgba(13,17,23,0.85)] backdrop-blur-lg">
        <div className="mx-auto flex h-14 max-w-[1200px] items-center justify-between px-5">
          <div className="flex items-center gap-6">
            <span className="text-base font-bold text-[#3ECF8E]">SemanticCache</span>
            {["Explorer", "Metrics", "Integrations", "Settings"].map((t) => (
              <button
                key={t}
                onClick={() => setActiveNav(t)}
                className={`hidden text-sm transition md:inline-block ${activeNav === t ? "nav-link-active font-medium" : "text-[#7d8590] hover:text-foreground"}`}
              >
                {t}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden text-sm text-[#7d8590] md:inline">Docs</span>
            <button className="rounded-lg border border-[#3ECF8E] bg-transparent px-3.5 py-1.5 text-sm font-semibold text-[#3ECF8E] transition hover:bg-[rgba(62,207,142,0.08)]">
              Deploy Node
            </button>
          </div>
        </div>
      </nav>

      <div className="mx-auto w-full max-w-[960px] px-5 py-10 md:py-14">
        <AnimatePresence mode="wait">
          {article ? (
            <ArticleView
              key="article"
              article={article}
              loading={articleLoading}
              onBack={backToResults}
              similarity={articleSimilarity}
            />
          ) : (
            <motion.div
              key="main"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {/* ═══════ HERO ═══════ */}
              <header className="mx-auto mb-10 max-w-[800px] text-center">
                <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-mono text-muted-foreground">
                  <Sparkles className="h-3.5 w-3.5 text-[#3ECF8E]" />
                  SemanticCache · v2.4.1
                </div>
                <h1 className="text-4xl font-bold leading-[1.1] tracking-tight md:text-6xl">
                  <span className="text-gradient-brand">Semantic Cache Search</span>
                </h1>
                <p className="mx-auto mt-4 max-w-[640px] text-base text-muted-foreground md:text-lg">
                  Fast, cluster-aware document retrieval with intelligent caching across 20,000 documents.
                </p>
              </header>

              {/* ═══════ SEARCH CONSOLE (matches: image (3).png) ═══════ */}
              <section className="glass-strong mb-8 rounded-2xl p-5 md:p-7">
                {/* title bar */}
                <div className="mb-5 flex items-center gap-2.5">
                  <div className="relative grid h-7 w-7 place-items-center rounded-md bg-[rgba(62,207,142,0.15)]">
                    <div className="h-3 w-3 rounded-full bg-[#3ECF8E] shadow-[0_0_10px_#3ECF8E]" />
                  </div>
                  <span className="text-base font-medium tracking-tight">Vector Search Engine</span>
                </div>

                {/* search input row */}
                <div
                  className={`group relative flex items-stretch gap-2 rounded-xl border border-[#30363d] bg-[#0d1117]/60 p-1.5 transition focus-within:border-[#3ECF8E]/60 focus-within:shadow-[0_0_0_3px_rgba(62,207,142,0.12)] ${inputError ? "ring-error-flash !border-[#f85149]" : ""}`}
                >
                  <div className="flex flex-1 items-center gap-2 px-3">
                    <Search className="h-4 w-4 text-[#7d8590]" />
                    <input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={onKey}
                      placeholder="Search across 20,000 documents..."
                      className="w-full bg-transparent py-2.5 text-[15px] text-foreground placeholder:text-[#484f58] focus:outline-none"
                    />
                  </div>
                  <button
                    onClick={handleSubmit}
                    disabled={loading}
                    className="btn-gradient-brand inline-flex items-center gap-1.5 rounded-lg px-5 py-2.5 text-sm font-semibold shadow-[0_4px_20px_rgba(62,207,142,0.25)] transition hover:brightness-110 active:scale-[0.98] disabled:opacity-60"
                  >
                    Search <ArrowRight className="h-4 w-4" />
                  </button>
                </div>

                {/* suggested query pills */}
                <div className="mt-5">
                  <div className="mb-2.5 font-mono text-[11px] uppercase tracking-[0.14em] text-[#7d8590]">
                    Suggested Queries
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {SUGGESTED.map((p) => (
                      <button
                        key={p}
                        onClick={() => onPill(p)}
                        className="rounded-full border border-[#30363d] bg-white/[0.02] px-3.5 py-1.5 text-[13px] text-foreground/85 transition hover:border-[#3ECF8E]/40 hover:bg-[rgba(62,207,142,0.06)] hover:text-foreground"
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                </div>

                {/* similarity threshold slider */}
                <div className="mt-6 border-t border-white/5 pt-5">
                  <div className="mb-3 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm font-medium">
                      <SlidersHorizontal className="h-4 w-4 text-[#7d8590]" />
                      Similarity Threshold
                    </div>
                    <div className="rounded-md border border-[#3ECF8E]/40 bg-[rgba(62,207,142,0.08)] px-2.5 py-1 font-mono text-sm font-bold text-[#3ECF8E]">
                      {threshold.toFixed(2)}
                    </div>
                  </div>
                  <input
                    type="range"
                    min={0.7}
                    max={0.98}
                    step={0.01}
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="slider-brand w-full"
                    style={{ "--slider-pct": `${sliderPct}%` } as React.CSSProperties}
                  />
                  <div className="mt-2 flex justify-between font-mono text-[11px] text-[#7d8590]">
                    <span>0.70 (Broad)</span>
                    <span>0.98 (Exact)</span>
                  </div>
                </div>
              </section>

              {/* ═══════ ERROR BANNER ═══════ */}
              {error && (
                <div className="mb-6 rounded-xl border border-[#d29922]/40 bg-[rgba(210,153,34,0.06)] p-4 text-sm text-[#d29922] backdrop-blur">
                  {error}
                </div>
              )}

              {/* ═══════ RESULTS / SKELETON / EMPTY ═══════ */}
              <section className="mb-12 min-h-[280px]">
                <AnimatePresence mode="wait">
                  {loading ? (
                    <SkeletonResults key="skel" />
                  ) : response ? (
                    <ResultsView
                      key="results"
                      response={response}
                      onOpen={openArticle}
                      responseTimeMs={responseTimeMs}
                    />
                  ) : (
                    <EmptyState key="empty" />
                  )}
                </AnimatePresence>
              </section>

              {/* ═══════ TELEMETRY (matches: image.png) ═══════ */}
              <Telemetry stats={stats} onClear={clearCache} />

              {/* ═══════ FOOTER ═══════ */}
              <footer className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-white/5 pt-5 font-mono text-[11px] text-[#7d8590]">
                <span>&copy; 2024 SemanticCache Inc. Protocol v2.4.1</span>
                <div className="flex gap-4">
                  <span className="cursor-pointer hover:text-foreground">Privacy</span>
                  <span className="cursor-pointer hover:text-foreground">Terms</span>
                  <span className="cursor-pointer hover:text-foreground">System Status</span>
                  <span className="cursor-pointer hover:text-foreground">Changelog</span>
                </div>
              </footer>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

/* ────────────────────────────────────────────── */
/*  EMPTY STATE                                    */
/* ────────────────────────────────────────────── */
function EmptyState() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col items-center justify-center gap-3 py-20 text-center text-[#7d8590]"
    >
      <div className="grid h-14 w-14 place-items-center rounded-full border border-[#30363d] bg-white/[0.03]">
        <Search className="h-6 w-6" />
      </div>
      <p className="text-sm">Enter a query to search</p>
    </motion.div>
  );
}

/* ────────────────────────────────────────────── */
/*  SKELETON LOADING                               */
/* ────────────────────────────────────────────── */
function SkeletonResults() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-3"
    >
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="glass animate-pulse rounded-xl p-6"
          style={{ animationDelay: `${i * 120}ms` }}
        >
          <div className="mb-3 flex justify-between">
            <div className="h-5 w-24 rounded bg-white/5" />
            <div className="h-5 w-16 rounded bg-white/5" />
          </div>
          <div className="mb-3 h-6 w-3/5 rounded bg-white/5" />
          <div className="mb-2 h-3 w-full rounded bg-white/5" />
          <div className="mb-1.5 h-3 w-5/6 rounded bg-white/5" />
          <div className="h-3 w-4/6 rounded bg-white/5" />
        </div>
      ))}
    </motion.div>
  );
}

/* ────────────────────────────────────────────── */
/*  RESULTS VIEW                                   */
/* ────────────────────────────────────────────── */
function ResultsView({
  response,
  onOpen,
  responseTimeMs,
}: {
  response: QueryResponse;
  onOpen: (r: ResultItem) => void;
  responseTimeMs: number | null;
}) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-4"
    >
      <CacheBanner response={response} responseTimeMs={responseTimeMs} />
      <div className="space-y-3">
        {response.results.map((r, i) => (
          <motion.button
            key={`${r.doc_id}-${i}`}
            type="button"
            onClick={() => onOpen(r)}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: i * 0.08, ease: "easeOut" }}
            className="glass group block w-full rounded-xl p-5 text-left transition-all hover:border-[#3ECF8E]/30 hover:shadow-[0_10px_40px_rgba(0,0,0,0.4)] hover:translate-y-[-2px]"
          >
            {/* top row: category + score */}
            <div className="mb-3 flex items-start justify-between gap-4">
              <span
                className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-xs ${categoryTint(r.category)}`}
              >
                <Folder className="h-3 w-3" />
                {r.category}
              </span>
              <span className="shrink-0 font-mono text-sm text-[#7d8590]">
                Score{" "}
                <span className="font-bold text-[#3ECF8E]">
                  {r.similarity_score.toFixed(2)}
                </span>
              </span>
            </div>

            {/* document title */}
            <h3 className="mb-2 text-xl font-bold tracking-tight">
              Document #{r.doc_id}
            </h3>

            {/* text snippet */}
            <p
              className="text-sm leading-relaxed text-foreground/70"
              style={{
                display: "-webkit-box",
                WebkitLineClamp: 3,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
              }}
            >
              {r.text}
            </p>

            {/* bottom row */}
            <div className="mt-4 flex items-center justify-between border-t border-white/5 pt-3">
              <span className="inline-flex items-center gap-1.5 font-mono text-xs text-[#7d8590]">
                <Database className="h-3.5 w-3.5" />
                Doc #{r.doc_id} &middot; Cluster {r.dominant_cluster}
              </span>
              <span className="inline-flex items-center gap-1 text-sm font-semibold text-[#3ECF8E] transition group-hover:gap-2">
                Read Full Article <ArrowRight className="h-3.5 w-3.5" />
              </span>
            </div>
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}

/* ────────────────────────────────────────────── */
/*  CACHE BANNER (matches: image (1).png HIT,      */
/*  image (4).png MISS)                            */
/* ────────────────────────────────────────────── */
function CacheBanner({
  response,
  responseTimeMs,
}: {
  response: QueryResponse;
  responseTimeMs: number | null;
}) {
  const hit = response.cache_hit;
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.25 }}
      className={`rounded-xl border p-4 backdrop-blur-sm ${
        hit
          ? "border-[#3ECF8E]/40 bg-[rgba(62,207,142,0.06)] glow-brand"
          : "border-[#f85149]/40 bg-[rgba(248,81,73,0.05)] glow-error"
      }`}
    >
      <div className="flex flex-wrap items-center gap-3">
        {/* dot + label */}
        <div className="flex items-center gap-2.5 pr-3">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              hit
                ? "bg-[#3ECF8E] shadow-[0_0_12px_#3ECF8E]"
                : "bg-[#f85149] shadow-[0_0_12px_#f85149]"
            }`}
          />
          <span
            className={`font-mono text-lg font-bold tracking-tight ${
              hit ? "text-[#3ECF8E]" : "text-[#f85149]"
            }`}
          >
            Cache {hit ? "HIT" : "MISS"}
          </span>
        </div>

        {/* divider */}
        <div className="h-6 w-px bg-white/10" />

        {/* center text */}
        <div className="min-w-0 flex-1">
          {hit ? (
            <div className="space-y-0.5">
              <div className="truncate text-sm text-foreground/85">
                <span className="text-[#7d8590]">Matched: </span>
                <span className="font-mono text-foreground/95">{response.matched_query}</span>
              </div>
              {responseTimeMs !== null && (
                <div className="inline-flex items-center gap-1 font-mono text-xs text-[#7d8590]">
                  <Timer className="h-3 w-3" />
                  Response time: {responseTimeMs}ms
                </div>
              )}
            </div>
          ) : (
            <span className="text-sm text-[#7d8590]">Performing vector search...</span>
          )}
        </div>

        {/* right badge */}
        <div
          className={`shrink-0 rounded-md border px-3 py-1.5 font-mono text-sm font-bold ${
            hit
              ? "border-[#3ECF8E]/40 bg-[rgba(62,207,142,0.10)] text-[#3ECF8E]"
              : "border-[#d29922]/40 bg-[rgba(210,153,34,0.08)] text-[#d29922]"
          }`}
        >
          {hit ? (
            <>
              Similarity{" "}
              <span className="ml-1">{(response.similarity_score ?? 0).toFixed(2)}</span>
            </>
          ) : (
            <span className="inline-flex items-center gap-1">
              <Timer className="h-3 w-3" />
              {responseTimeMs ?? 185}ms
            </span>
          )}
        </div>
      </div>
    </motion.div>
  );
}

/* ────────────────────────────────────────────── */
/*  ARTICLE VIEW (matches: image (5).png)          */
/* ────────────────────────────────────────────── */
function ArticleView({
  article,
  loading,
  onBack,
  similarity,
}: {
  article: FullDoc;
  loading: boolean;
  onBack: () => void;
  similarity: number | null;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ duration: 0.3 }}
    >
      <button
        onClick={onBack}
        className="mb-5 inline-flex items-center gap-2 text-sm text-[#7d8590] transition hover:text-[#3ECF8E]"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Results
      </button>

      <div className="glass-strong grid gap-6 rounded-2xl p-6 md:p-8 lg:grid-cols-[1fr_300px]">
        {/* main content */}
        <div>
          {/* badges */}
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 font-mono text-xs uppercase ${categoryTint(article.category)}`}
            >
              <Folder className="h-3 w-3" />
              {article.category}
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-md border border-[#3ECF8E]/30 bg-[rgba(62,207,142,0.06)] px-2.5 py-1 font-mono text-xs text-[#3ECF8E]">
              Cluster {article.dominant_cluster}
            </span>
          </div>

          {/* title */}
          <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
            Document #{article.doc_id}
          </h2>

          {/* divider */}
          <div className="my-5 h-px bg-white/[0.06]" />

          {/* body */}
          {loading ? (
            <div className="space-y-2.5">
              {Array.from({ length: 8 }).map((_, i) => (
                <div
                  key={i}
                  className="h-3.5 animate-pulse rounded bg-white/5"
                  style={{ width: `${70 + (i % 4) * 8}%` }}
                />
              ))}
            </div>
          ) : (
            <article
              className="max-w-[700px] whitespace-pre-wrap text-[15px] text-foreground/85"
              style={{ lineHeight: 1.75 }}
            >
              {article.clean_text}
            </article>
          )}
        </div>

        {/* ── sidebar metadata (matches right column in image (5).png) ── */}
        <aside className="space-y-4 lg:border-l lg:border-white/[0.06] lg:pl-6">
          <div className="flex items-center gap-2 text-sm">
            <BarChart3 className="h-4 w-4 text-[#3ECF8E]" />
            <span className="font-semibold">Analysis Metadata</span>
          </div>

          {/* similarity score */}
          <div className="rounded-lg border border-[#30363d] bg-[#0d1117]/50 p-4">
            <div className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-[#7d8590]">
              Similarity Score
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className="font-mono text-2xl font-bold text-[#3ECF8E]">
                {(similarity ?? 0).toFixed(2)}
              </span>
              <span className="font-mono text-xs text-[#7d8590]">cosine</span>
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/5">
              <div
                className="h-full rounded-full bg-[#3ECF8E]"
                style={{ width: `${Math.min(100, (similarity ?? 0) * 100)}%` }}
              />
            </div>
          </div>

          {/* category match */}
          <div className="rounded-lg border border-[#30363d] bg-[#0d1117]/50 p-4">
            <div className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-[#7d8590]">
              Category Match
            </div>
            <div className="font-mono text-base font-semibold">{article.category}</div>
            <div className="mt-1.5 inline-flex items-center gap-1 text-xs text-[#3ECF8E]">
              <span className="text-[#3ECF8E]">&#10003;</span> High Confidence
            </div>
          </div>

          {/* cluster */}
          <div className="rounded-lg border border-[#30363d] bg-[#0d1117]/50 p-4">
            <div className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-[#7d8590]">
              Dominant Cluster
            </div>
            <div className="flex items-center gap-2 font-mono text-base font-semibold">
              <span className="h-2.5 w-2.5 rounded-full bg-[#3ECF8E]" />
              Cluster {article.dominant_cluster}
            </div>
            <div className="mt-1.5 font-mono text-[11px] text-[#7d8590]">
              Centroid Dist: 0.12
            </div>
          </div>

          {/* export button */}
          <button className="flex w-full items-center justify-center gap-2 rounded-lg border border-[#30363d] bg-[#161b22] py-2.5 text-sm font-medium text-foreground/85 transition hover:border-white/20 hover:bg-[#1c2128]">
            <Download className="h-4 w-4" />
            Export JSON
          </button>
        </aside>
      </div>
    </motion.div>
  );
}

/* ────────────────────────────────────────────── */
/*  COUNT-UP HOOK                                  */
/* ────────────────────────────────────────────── */
function useCountUp(value: number, duration = 600) {
  const [n, setN] = useState(value);
  const fromRef = useRef(value);
  useEffect(() => {
    const from = fromRef.current;
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setN(from + (value - from) * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
      else fromRef.current = value;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);
  return n;
}

/* ────────────────────────────────────────────── */
/*  TELEMETRY (matches: image.png)                 */
/* ────────────────────────────────────────────── */
function Telemetry({
  stats,
  onClear,
}: {
  stats: CacheStats | null;
  onClear: () => void;
}) {
  const hitRate = useCountUp(stats ? stats.hit_rate * 100 : 0);
  const entries = useCountUp(stats?.total_entries ?? 0);
  const hits = useCountUp(stats?.hit_count ?? 0);
  const misses = useCountUp(stats?.miss_count ?? 0);

  const total = (stats?.hit_count ?? 0) + (stats?.miss_count ?? 0);
  const hitsPct = total ? ((stats?.hit_count ?? 0) / total) * 100 : 50;

  // circular progress
  const R = 22;
  const C = 2 * Math.PI * R;
  const offset = C - (hitRate / 100) * C;

  return (
    <section>
      <div className="mb-4 flex items-end justify-between">
        <div>
          <h2 className="text-xl font-bold tracking-tight">Telemetry Overview</h2>
          <p className="text-xs text-[#7d8590]">
            Real-time metrics for Edge_Cluster_01 cache performance.
          </p>
        </div>
        <button
          className="text-[#7d8590] transition hover:text-foreground"
          onClick={() => {
            /* would refresh */
          }}
        >
          <RefreshCw className="h-4 w-4" />
        </button>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">
        {/* ── Hit Rate ── */}
        <div className="glass rounded-xl p-4">
          <div className="mb-2 flex items-center justify-between text-sm text-[#7d8590]">
            Hit Rate <BarChart3 className="h-4 w-4" />
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-baseline gap-0.5">
              <span className="text-3xl font-bold tracking-tight">
                {Math.round(hitRate)}
              </span>
              <span className="text-lg text-[#7d8590]">%</span>
            </div>
            <svg width="52" height="52" viewBox="0 0 52 52" className="-rotate-90">
              <circle
                cx="26"
                cy="26"
                r={R}
                stroke="rgba(255,255,255,0.06)"
                strokeWidth="4"
                fill="none"
              />
              <circle
                cx="26"
                cy="26"
                r={R}
                stroke="#3ECF8E"
                strokeWidth="4"
                fill="none"
                strokeLinecap="round"
                strokeDasharray={C}
                strokeDashoffset={offset}
                style={{ transition: "stroke-dashoffset 0.6s ease" }}
              />
            </svg>
          </div>
          <div className="mt-3 font-mono text-[11px] text-[#3ECF8E]">
            &uarr; live
          </div>
        </div>

        {/* ── Cache Entries ── */}
        <div className="glass rounded-xl p-4">
          <div className="mb-2 flex items-center justify-between text-sm text-[#7d8590]">
            Cache Entries <ListChecks className="h-4 w-4" />
          </div>
          <div className="flex items-baseline gap-0.5">
            <span className="text-3xl font-bold tracking-tight">
              {Math.round(entries)}
            </span>
            <span className="text-lg text-[#7d8590]">k</span>
          </div>
          <div className="mt-3 text-xs text-[#7d8590]">Total active items stored</div>
        </div>

        {/* ── Hits / Misses ── */}
        <div className="glass rounded-xl p-4">
          <div className="mb-2 flex items-center justify-between text-sm text-[#7d8590]">
            Hits / Misses <ArrowLeftRight className="h-4 w-4" />
          </div>
          <div className="flex items-baseline gap-1.5 text-3xl font-bold tracking-tight">
            <span className="text-[#3ECF8E]">{Math.round(hits)}</span>
            <span className="text-base text-[#7d8590]">/</span>
            <span className="text-[#f85149]">{Math.round(misses)}</span>
          </div>
          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[#f85149]/30">
            <div
              className="h-full bg-[#3ECF8E] transition-all duration-700"
              style={{ width: `${hitsPct}%` }}
            />
          </div>
        </div>

        {/* ── Clear Cache ── */}
        <button
          onClick={onClear}
          className="group rounded-xl border border-[#f85149]/25 bg-[rgba(248,81,73,0.04)] p-4 text-left transition hover:border-[#f85149]/50 hover:bg-[rgba(248,81,73,0.08)]"
        >
          <div className="flex h-full flex-col items-center justify-center gap-2 text-[#f85149]">
            <Trash2 className="h-6 w-6 transition group-hover:scale-110" />
            <span className="text-sm font-semibold">Clear Cache</span>
          </div>
        </button>
      </div>

      {/* ── Recent Cache Operations table (matches: image.png) ── */}
      <div className="glass mt-4 rounded-xl p-5">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-base font-semibold">Recent Cache Operations</h3>
          <RefreshCw className="h-4 w-4 text-[#7d8590]" />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/[0.06] text-left text-xs text-[#7d8590]">
                <th className="pb-3 font-medium">Timestamp</th>
                <th className="pb-3 font-medium">Key Prefix</th>
                <th className="pb-3 font-medium">Action</th>
                <th className="pb-3 font-medium">Duration</th>
              </tr>
            </thead>
            <tbody className="font-mono text-xs">
              <tr className="border-b border-white/[0.03]">
                <td className="py-3 text-[#7d8590]">10:42:01.002Z</td>
                <td className="py-3 text-[#3ECF8E]">user:profile:*</td>
                <td className="py-3">
                  <span className="rounded border border-[#3ECF8E]/30 bg-[rgba(62,207,142,0.08)] px-1.5 py-0.5 text-[#3ECF8E]">
                    HIT
                  </span>
                </td>
                <td className="py-3 text-[#7d8590]">12ms</td>
              </tr>
              <tr className="border-b border-white/[0.03]">
                <td className="py-3 text-[#7d8590]">10:41:55.120Z</td>
                <td className="py-3 text-[#3ECF8E]">org:settings:192</td>
                <td className="py-3">
                  <span className="rounded border border-[#f85149]/30 bg-[rgba(248,81,73,0.08)] px-1.5 py-0.5 text-[#f85149]">
                    MISS
                  </span>
                </td>
                <td className="py-3 text-[#7d8590]">84ms</td>
              </tr>
              <tr>
                <td className="py-3 text-[#7d8590]">10:41:50.005Z</td>
                <td className="py-3 text-[#3ECF8E]">query:vector:abc...</td>
                <td className="py-3">
                  <span className="rounded border border-[#3ECF8E]/30 bg-[rgba(62,207,142,0.08)] px-1.5 py-0.5 text-[#3ECF8E]">
                    HIT
                  </span>
                </td>
                <td className="py-3 text-[#7d8590]">8ms</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
