import numpy as np
from sklearn.mixture import GaussianMixture


class ClusterPredictor:

    def __init__(self):

        embeddings = np.load("data/embeddings/document_embeddings.npy")

        # GMM (Gaussian Mixture Model) is chosen explicitly over KMeans because 
        # the assignment forbids hard cluster assignments. Documents might span 
        # multiple topics (e.g. politics & religion), so GMM lets us extract soft 
        # probabilities for each cluster to reflect the actual messiness of the text.
        self.gmm = GaussianMixture(n_components=20, covariance_type="diag", random_state=42)

        self.gmm.fit(embeddings)

    def predict_cluster(self, query_embedding):

        probs = self.gmm.predict_proba(query_embedding)

        cluster_id = probs.argmax()

        return int(cluster_id)