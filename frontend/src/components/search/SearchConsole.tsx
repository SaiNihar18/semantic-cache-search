import { useEffect, useRef, useState, type CSSProperties, type FormEvent } from "react";
import { ArrowRight, ChevronDown, Loader2, Search, SlidersHorizontal } from "lucide-react";

export const THRESHOLD_MIN = 0.7;
export const THRESHOLD_MAX = 0.98;

const EXAMPLES = [
  "How do rockets work?",
  "How does a rocket work?",
  "Best graphics cards for gaming",
  "Which graphics card is best for gaming?",
  "Treatment for migraines",
];

type Props = {
  query: string;
  threshold: number;
  loading: boolean;
  onSubmit: (query: string, threshold: number) => void;
};

export function SearchConsole({ query, threshold, loading, onSubmit }: Props) {
  const [text, setText] = useState(query);
  const [t, setT] = useState(threshold);
  const [open, setOpen] = useState(false);
  const [invalid, setInvalid] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Keep the form in sync when the URL changes (Back/Forward, shared links).
  useEffect(() => setText(query), [query]);
  useEffect(() => setT(threshold), [threshold]);

  // "/" focuses the search box.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (e.key === "/" && tag !== "INPUT" && tag !== "TEXTAREA") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function submit(q: string) {
    const trimmed = q.trim();
    if (!trimmed) {
      setInvalid(true);
      setTimeout(() => setInvalid(false), 600);
      inputRef.current?.focus();
      return;
    }
    if (loading) return;
    onSubmit(trimmed, t);
  }

  function handleForm(e: FormEvent) {
    e.preventDefault();
    submit(text);
  }

  const pct = ((t - THRESHOLD_MIN) / (THRESHOLD_MAX - THRESHOLD_MIN)) * 100;

  return (
    <section aria-label="Search" className="panel p-4 sm:p-5">
      <form onSubmit={handleForm} role="search">
        <div
          className={`flex items-center gap-2 rounded-lg border border-border bg-background p-1.5 transition focus-within:border-brand focus-within:ring-2 focus-within:ring-brand/20 ${
            invalid ? "ring-error-flash !border-destructive" : ""
          }`}
        >
          <Search className="ml-2.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
          <input
            ref={inputRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            maxLength={500}
            aria-label="Search query"
            placeholder="Search across 20,000 newsgroup documents"
            className="h-10 min-w-0 flex-1 bg-transparent text-[15px] text-foreground outline-none placeholder:text-muted-foreground/70"
          />
          <kbd className="hidden rounded border border-border px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground sm:block">
            /
          </kbd>
          <button
            type="submit"
            disabled={loading}
            className="inline-flex h-10 items-center gap-2 rounded-md bg-brand px-4 text-sm font-semibold text-brand-foreground transition hover:bg-brand/90 disabled:opacity-60"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            ) : (
              <>
                Search <ArrowRight className="h-4 w-4" aria-hidden />
              </>
            )}
            {loading && <span>Searching</span>}
          </button>
        </div>
      </form>

      <div className="mt-4">
        <div className="label-caps mb-2">Try an example</div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => {
                setText(p);
                submit(p);
              }}
              className="rounded-full border border-border bg-secondary/50 px-3 py-1 text-[13px] text-foreground/85 transition hover:border-brand/50 hover:text-foreground"
            >
              {p}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Run a pair back to back (rockets, graphics cards) to watch the cache recognise a paraphrase.
        </p>
      </div>

      <div className="mt-4 border-t border-border pt-3">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-controls="advanced-panel"
          className="flex w-full items-center justify-between text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          <span className="inline-flex items-center gap-2">
            <SlidersHorizontal className="h-4 w-4" aria-hidden />
            Advanced: cache similarity threshold
          </span>
          <span className="inline-flex items-center gap-2">
            <span className="rounded-md border border-brand/40 bg-brand/10 px-2 py-0.5 font-mono text-xs font-semibold text-brand">
              {t.toFixed(2)}
            </span>
            <ChevronDown
              className={`h-4 w-4 transition-transform ${open ? "rotate-180" : ""}`}
              aria-hidden
            />
          </span>
        </button>
        {open && (
          <div id="advanced-panel" className="mt-4">
            <input
              type="range"
              aria-label="Similarity threshold"
              min={THRESHOLD_MIN}
              max={THRESHOLD_MAX}
              step={0.01}
              value={t}
              onChange={(e) => setT(parseFloat(e.target.value))}
              className="slider-brand"
              style={{ "--slider-pct": `${pct}%` } as CSSProperties}
            />
            <div className="mt-2 flex justify-between font-mono text-[11px] text-muted-foreground">
              <span>0.70 Broad</span>
              <span>0.85 Recommended</span>
              <span>0.98 Exact</span>
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              Controls how close a new query must be to a previous one to be answered from cache. It
              does not change which documents are returned on a miss.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
