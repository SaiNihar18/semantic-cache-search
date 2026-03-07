import numpy as np
from collections import defaultdict


class SemanticCache:

    def __init__(self, similarity_threshold=0.85):

        # We structure the cache as a dictionary of lists grouped by cluster_id.
        # This "cluster-aware caching" restricts the search domain. Instead of 
        # doing an O(N) scan across all cached queries, we only search within the 
        # predicted semantic zone. This keeps lookup latency scaling flat.
        self.cache = defaultdict(list)

        # The core tunable parameter. This threshold determines the boundary between 
        # different concepts. We use 0.85 as a balanced sweet spot (as values too low
        # like 0.70 trigger false positives, while 0.95 acts almost like an exact string match).
        self.similarity_threshold = similarity_threshold


    def lookup(self, query_embedding, cluster_id):

        candidates = self.cache.get(cluster_id, [])

        best_match = None
        best_score = -1

        for entry in candidates:

            score = float(np.dot(query_embedding, entry["embedding"].T)[0][0])

            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= self.similarity_threshold:

            return {
                "cache_hit": True,
                "matched_query": best_match["query"],
                "similarity_score": float(best_score),
                "result": best_match["result"],
                "dominant_cluster": int(cluster_id)
            }

        return {"cache_hit": False}


    def add_to_cache(self, query, query_embedding, result, cluster_id):

        for entry in self.cache.get(cluster_id, []):
            if entry["query"] == query:
                return

        entry = {
            "query": query,
            "embedding": query_embedding,
            "result": result
        }

        self.cache[cluster_id].append(entry)


    def stats(self):

        total_entries = sum(len(v) for v in self.cache.values())

        return {
            "total_entries": total_entries
        }


    def clear(self):

        self.cache.clear()