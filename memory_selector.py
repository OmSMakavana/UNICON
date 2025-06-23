# memory_selector.py
import numpy as np
from sklearn.cluster import KMeans

class KMeansMemorySelector:
    def __init__(self, num_clusters=100):
        self.num_clusters = num_clusters
        self.features = []
        self.labels = []
        self.indices = []  # New list to track original dataset indices

    def reset(self):
        self.features = []
        self.labels = []
        self.indices = []  # Reset indices

    def collect(self, feats, targets, indices=None):
        self.features.extend(feats)
        self.labels.extend(targets)
        if indices is not None:
            self.indices.extend(indices)
        else:
            # If indices not provided, use sequential indices
            self.indices.extend(range(len(self.features) - len(feats), len(self.features)))

    def select(self):
        if len(self.features) < self.num_clusters:
            print("[KMeans] Not enough data to cluster.")
            if len(self.features) < 100:  # arbitrary minimum
                print("[KMeans] Too few samples, falling back to JSD-based selection")
                return None, None, None
            return self.features, self.labels, np.array(self.indices)  # Convert to numpy array

        X = np.array(self.features)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_

        selected_indices = []
        for center in centers:
            distances = np.linalg.norm(X - center, axis=1)
            selected_indices.append(np.argmin(distances))

        selected_features = [self.features[i] for i in selected_indices]
        selected_labels = [self.labels[i] for i in selected_indices]
        selected_original_indices = np.array([self.indices[i] for i in selected_indices])  # Convert to numpy array

        # Additional validation
        if len(selected_original_indices) < 100:
            print("[KMeans] Selected too few samples, falling back to JSD-based selection")
            return None, None, None

        return selected_features, selected_labels, selected_original_indices
