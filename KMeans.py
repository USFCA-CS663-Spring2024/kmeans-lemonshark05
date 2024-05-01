import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as KMeans_sklearn
from sklearn.metrics import silhouette_score, v_measure_score

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
            # Define a small tolerance level for convergence check
            tolerance = 1e-4

            for point in x:
                distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
                if self.balanced:
                    # Adjust distances for balance
                    for i in range(self.k):
                        distances[i] *= (1 + cluster_list[i] / len(x))

                closest_centroid = np.argmin(distances)
                clusters[closest_centroid].append(point)
                labels.append(closest_centroid)

            # Update centroids
            new_centroids = []
            for i, cluster in enumerate(clusters):
                if len(cluster) > 0:
                    # Calculate the mean of the cluster if it has any points
                    mean_centroid = np.mean(cluster, axis=0)
                else:
                    # If the cluster has no points, retain the old centroid
                    mean_centroid = self.centroids[i]
                new_centroids.append(mean_centroid)
            centroid_shifts = np.linalg.norm(np.array(self.centroids) - np.array(new_centroids), axis=1)
            # Check convergence based on centroid shifts
            if np.all(centroid_shifts < tolerance):
                break
            self.centroids = new_centroids

        return labels, self.centroids


def evaluate_clusters(true_labels, predicted_labels):
    silhouette_avg = silhouette_score(X, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    return silhouette_avg, v_measure

if __name__ == "__main__":
    X, cluster_assignments = make_blobs(n_samples=700, centers=4, cluster_std=0.60, random_state=0)
    custom_kmeans = KMeans(k=4)
    predicted_labels_custom, centroids_custom = custom_kmeans.fit(X)

    # Evaluate custom KMeans
    silhouette_custom, v_measure_custom = evaluate_clusters(cluster_assignments, predicted_labels_custom)

    # Using scikit-learn KMeans
    kmeans_sklearn = KMeans_sklearn(n_clusters=4, random_state=0)
    kmeans_sklearn.fit(X)
    predicted_labels_sklearn = kmeans_sklearn.labels_
    silhouette_sklearn, v_measure_sklearn = evaluate_clusters(cluster_assignments, predicted_labels_sklearn)

    print(f"Custom KMeans: Silhouette Score = {silhouette_custom}, V-measure Score = {v_measure_custom}")
    print(f"Scikit-learn KMeans: Silhouette Score = {silhouette_sklearn}, V-measure Score = {v_measure_sklearn}")