from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingModel:

    def __init__(self):

        # Using sentence-transformers/all-MiniLM-L6-v2 here. 
        # It's a lightweight, 384-dimensional model that's incredibly fast for inference.
        # Real-time API constraints mean we can't afford massive models like BERT-large,
        # and MiniLM gives more than enough semantic density for our exact dataset size.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text):

        embedding = self.model.encode([text], convert_to_numpy=True)

        faiss.normalize_L2(embedding)

        return embedding