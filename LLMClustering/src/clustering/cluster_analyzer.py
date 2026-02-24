from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

class ClusterAnalyzer:
    def __init__(self):
        self.model = None
        self.labels_ = None
        
    def kmeans_clustering(self, embeddings, n_clusters=5):
        """
        Perform K-means clustering on embeddings.
        
        Args:
            embeddings (np.ndarray): Array of embeddings
            n_clusters (int): Number of clusters
            
        Returns:
            np.ndarray: Cluster labels
        """
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels_ = self.model.fit_predict(embeddings)
        return self.labels_
    
    def dbscan_clustering(self, embeddings, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering on embeddings.
        
        Args:
            embeddings (np.ndarray): Array of embeddings
            eps (float): Maximum distance between samples
            min_samples (int): Minimum number of samples in a cluster
            
        Returns:
            np.ndarray: Cluster labels
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = self.model.fit_predict(embeddings)
        return self.labels_
    
    def evaluate_clustering(self, embeddings):
        """
        Evaluate clustering quality using silhouette score.
        
        Args:
            embeddings (np.ndarray): Array of embeddings
            
        Returns:
            float: Silhouette score
        """
        if self.labels_ is None:
            raise ValueError("Must perform clustering first")
        
        return silhouette_score(embeddings, self.labels_)