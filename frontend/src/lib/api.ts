export const API = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";
export const API_DOCS = `${API}/docs`;
export const REPO_URL = "https://github.com/SaiNihar18/semantic-cache-search";

export type ResultItem = {
  doc_id: number;
  category: string;
  text: string;
  similarity_score: number;
  dominant_cluster: number;
};

export type QueryResponse = {
  query: string;
  cache_hit: boolean;
  matched_query: string | null;
  similarity_score: number | null;
  result: string;
  results: ResultItem[];
  dominant_cluster: number;
};

export type SearchResult = QueryResponse & { elapsedMs: number };

export type CacheStats = {
  total_entries: number;
  hit_count: number;
  miss_count: number;
  hit_rate: number;
};

export type FullDoc = {
  doc_id: number;
  category: string;
  clean_text: string;
  dominant_cluster: number;
};

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, init);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(`Could not reach the API at ${API}. It may be offline or waking up.`);
  }
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : "";
    } catch {
      /* ignore */
    }
    throw new ApiError(detail || `The API returned an error (${res.status}).`, res.status);
  }
  return res.json() as Promise<T>;
}

export async function searchDocs(
  query: string,
  threshold: number,
  signal?: AbortSignal,
): Promise<SearchResult> {
  const start = performance.now();
  const data = await request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, similarity_threshold: threshold }),
    signal,
  });
  return { ...data, elapsedMs: Math.round(performance.now() - start) };
}

export const getStats = (signal?: AbortSignal) => request<CacheStats>("/cache/stats", { signal });
export const getDocument = (id: number, signal?: AbortSignal) =>
  request<FullDoc>(`/document/${id}`, { signal });
export const clearCacheApi = () => request<{ message: string }>("/cache", { method: "DELETE" });

/** First sentence-ish line of a post, used as a card title. */
export function deriveTitle(text: string, max = 80): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return "Untitled post";
  const sentence = clean.split(/(?<=[.!?])\s/)[0];
  return sentence.length > max ? `${sentence.slice(0, max).trimEnd()}…` : sentence;
}
