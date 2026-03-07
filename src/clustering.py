import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
import os


class ClusterPredictor:

    def __init__(self):
        
        # Free Tier constraints (like Render's 512MB RAM limit) prevent us from training 
        # the GMM every single time the server boots up. The embeddings file is large
        # (~29.2 MiB) and calculating covariance matrix runs out of memory rapidly.
        # Instead, we load a pre-trained instance of the model.

        model_path = "data/embeddings/gmm_model.pkl"
        
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.gmm = pickle.load(f)
        else:
            print("GMM model not found! Training on the fly...")
            embeddings = np.load("data/embeddings/document_embeddings.npy")
            
            # GMM (Gaussian Mixture Model) is chosen explicitly over KMeans because 
            # the assignment forbids hard cluster assignments. Documents might span 
            # multiple topics (e.g. politics & religion), so GMM lets us extract soft 
            # probabilities for each cluster to reflect the actual messiness of the text.
            self.gmm = GaussianMixture(n_components=20, covariance_type="diag", random_state=42)
            self.gmm.fit(embeddings)
            
            # Save for future use
            with open(model_path, "wb") as f:
                pickle.dump(self.gmm, f)

    def predict_cluster(self, query_embedding):

        probs = self.gmm.predict_proba(query_embedding)

        cluster_id = probs.argmax()

        return int(cluster_id)