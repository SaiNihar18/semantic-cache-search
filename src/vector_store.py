import faiss
import numpy as np
import pandas as pd


class VectorStore:

    def __init__(self):

        # We use FAISS (IndexFlatIP) because inner product on L2-normalized vectors 
        # calculates exact cosine similarity efficiently.
        # Even though our dataset is ~20k docs, FAISS scales perfectly if the corpus 
        # grows to millions of posts without needing a heavyweight vector database.
        self.index = faiss.read_index("data/embeddings/faiss_index.bin")

        self.df = pd.read_csv("data/processed/newsgroups_clustered.csv")

    def search(self, query_embedding, top_k=5):

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            doc = self.df.iloc[idx]

            results.append({
                "doc_id": int(doc["doc_id"]),
                "category": str(doc["category"]),
                "text": str(doc["clean_text"]),
                "similarity_score": float(score),
                "dominant_cluster": int(doc["dominant_cluster"])
            })

        return results