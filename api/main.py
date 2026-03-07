from fastapi import FastAPI
from pydantic import BaseModel

from src.semantic_cache import SemanticCache
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.clustering import ClusterPredictor


app = FastAPI()

cache = SemanticCache()

embedding_model = EmbeddingModel()

vector_store = VectorStore()

cluster_model = ClusterPredictor()


class QueryRequest(BaseModel):

    query: str


hit_count = 0
miss_count = 0


@app.post("/query")
def query_api(request: QueryRequest):

    global hit_count, miss_count

    query = request.query

    query_embedding = embedding_model.embed(query)

    cluster_id = int(cluster_model.predict_cluster(query_embedding))

    cache_result = cache.lookup(query_embedding, cluster_id)

    if cache_result["cache_hit"]:

        hit_count += 1

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cache_result.get("matched_query"),
            "similarity_score": cache_result.get("similarity_score"),
            "result": cache_result.get("result"),
            "dominant_cluster": cluster_id
        }

    search_results = vector_store.search(query_embedding)

    result_text = str(search_results[0]["text"][:500])

    cache.add_to_cache(query, query_embedding, result_text, cluster_id)

    miss_count += 1

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result_text,
        "dominant_cluster": cluster_id
    }


@app.get("/cache/stats")
def cache_stats():

    total = hit_count + miss_count

    hit_rate = hit_count / total if total > 0 else 0

    return {
        "total_entries": cache.stats()["total_entries"],
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": hit_rate
    }


@app.delete("/cache")
def clear_cache():

    global hit_count, miss_count

    cache.clear()

    hit_count = 0
    miss_count = 0

    return {"message": "cache cleared"}