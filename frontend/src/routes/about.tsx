import { createFileRoute } from "@tanstack/react-router";
import { Cpu, Layers, Search, Zap } from "lucide-react";
import { GROUP_META, NEWSGROUPS } from "@/lib/categories";

export const Route = createFileRoute("/about")({
  head: () => ({ meta: [{ title: "About Data - SemanticCache" }] }),
  component: About,
});

const STATS = [
  { label: "Documents", value: "~20,000" },
  { label: "Categories", value: "20" },
  { label: "Embedding dims", value: "384" },
  { label: "GMM clusters", value: "20" },
];

const STEPS = [
  {
    title: "Encode",
    icon: Cpu,
    desc: "The query becomes a 384-dimensional vector using all-MiniLM-L6-v2.",
  },
  {
    title: "Cluster",
    icon: Layers,
    desc: "A Gaussian Mixture Model predicts the query's topic cluster.",
  },
  {
    title: "Cache lookup",
    icon: Zap,
    desc: "Only that cluster's cached queries are compared. A score at or above the threshold is a hit.",
  },
  {
    title: "FAISS search",
    icon: Search,
    desc: "On a miss, FAISS runs exact inner-product search across all document embeddings.",
  },
];

const MODEL = [
  ["Embedding model", "all-MiniLM-L6-v2", "22M parameters, fast CPU inference"],
  ["Vector index", "FAISS IndexFlatIP", "Exact search on L2-normalised vectors"],
  ["Clustering", "Gaussian Mixture Model", "20 components, soft assignment"],
];

const THRESHOLDS = [
  ["0.70", "Broad", "High hit rate. Approximate matches are acceptable."],
  ["0.85", "Recommended", "Balanced precision and recall for general use."],
  ["0.95+", "Exact", "Strict matching with a low hit rate."],
];

function About() {
  return (
    <div className="animate-fade-in space-y-10">
      <header className="max-w-[640px]">
        <h1 className="text-[28px] font-semibold tracking-tight sm:text-[32px]">About the data</h1>
        <p className="mt-3 text-muted-foreground">
          This engine indexes the 20 Newsgroups dataset, a classic NLP benchmark of roughly 20,000
          newsgroup posts across 20 topics from the early 1990s.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {STATS.map((s) => (
          <div key={s.label} className="panel p-4">
            <div className="label-caps">{s.label}</div>
            <div className="mt-2 text-2xl font-semibold tracking-tight">{s.value}</div>
          </div>
        ))}
      </div>

      <section aria-labelledby="pipeline">
        <h2 id="pipeline" className="text-lg font-semibold">
          How a query is processed
        </h2>
        <ol className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((s, i) => (
            <li key={s.title} className="panel p-4">
              <div className="flex items-center gap-3">
                <span className="grid h-8 w-8 place-items-center rounded-md bg-brand/15 text-brand">
                  <s.icon className="h-4 w-4" aria-hidden />
                </span>
                <span className="font-mono text-xs text-muted-foreground">Step {i + 1}</span>
              </div>
              <div className="mt-3 font-semibold">{s.title}</div>
              <p className="mt-1 text-sm text-muted-foreground">{s.desc}</p>
            </li>
          ))}
        </ol>
      </section>

      <div className="grid gap-4 md:grid-cols-2">
        <section className="panel p-5" aria-labelledby="model">
          <h2 id="model" className="text-lg font-semibold">
            Model and index
          </h2>
          <dl className="mt-4 space-y-3">
            {MODEL.map(([k, v, sub]) => (
              <div key={k} className="panel-inset p-3">
                <dt className="label-caps">{k}</dt>
                <dd className="mt-1 font-mono text-sm text-brand">{v}</dd>
                <dd className="text-xs text-muted-foreground">{sub}</dd>
              </div>
            ))}
          </dl>
        </section>
        <section className="panel p-5" aria-labelledby="thresholds">
          <h2 id="thresholds" className="text-lg font-semibold">
            Choosing a threshold
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            How close a new query must be to a cached one to be served from cache.
          </p>
          <ul className="mt-4 space-y-3">
            {THRESHOLDS.map(([v, label, desc]) => (
              <li key={v} className="panel-inset p-3">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm font-semibold">{v}</span>
                  <span className="text-xs text-brand">{label}</span>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{desc}</p>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <section className="panel p-5" aria-labelledby="cats">
        <h2 id="cats" className="text-lg font-semibold">
          Category breakdown
        </h2>
        <div className="mt-5 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {NEWSGROUPS.map((g) => (
            <div key={g.group}>
              <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                <span className={`h-2 w-2 rounded-full ${GROUP_META[g.group].dot}`} aria-hidden />
                {GROUP_META[g.group].label}
              </div>
              <ul className="divide-y divide-border rounded-lg border border-border">
                {g.items.map((it) => (
                  <li key={it.name} className="px-3 py-2">
                    <div className="font-mono text-xs">{it.name}</div>
                    <div className="text-xs text-muted-foreground">{it.desc}</div>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
