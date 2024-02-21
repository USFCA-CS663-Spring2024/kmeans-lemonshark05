import numpy as np


class KMeans:
    def __init__(self, k=5, max_iterations=100, balanced=False):
        # The target number of cluster centroids
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []
        self.balanced = balanced

    def fit(self, x):
        # Initialize the centroids by k
        self.centroids = x[np.random.choice(range(len(x)), self.k, replace=False)]
        cluster_list = [0] * self.k

        for _ in range(self.max_iterations):
            clusters = [[] for _ in range(self.k)]
            labels = []

            for point in x:
                distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
                if self.balanced:
                    for i in range(self.k):
                        distances[i] *= (1 + cluster_list[i] / len(x))

                closest_centroid = np.argmin(distances)
                clusters[closest_centroid].append(point)
                labels.append(closest_centroid)

            # Update centroids
            new_centroids = []
            for cluster in clusters:
                new_centroids.append(np.mean(cluster, axis=0))
            if np.array_equal(self.centroids, new_centroids):
                break
            else:
                self.centroids = new_centroids

        return labels, self.centroids
