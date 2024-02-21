import unittest
import numpy as np
from KMeans import KMeans

class TestKMeans(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0, 0], [2, 2], [0, 2], [2, 0],
            [10, 10], [8, 8], [10, 8], [8, 10]
        ])
        self.kmeans = KMeans(k=2, max_iterations=100)

    def test_fit(self):
        labels, centroids = self.kmeans.fit(self.X)

        self.assertEqual(len(np.unique(labels)), 2)

        first_cluster_label = labels[0]
        for label in labels[1:4]:
            self.assertEqual(label, first_cluster_label)

        second_cluster_label = labels[4]
        for label in labels[5:]:
            self.assertEqual(label, second_cluster_label)
        self.assertNotEqual(first_cluster_label, second_cluster_label)

        expected_centroids = np.array([[1, 1], [9, 9]])
        np.testing.assert_almost_equal(np.sort(centroids, axis=0), np.sort(expected_centroids, axis=0))

        print("\nCluster assignments:", labels)
        print("\nCalculated centroids:", centroids)

    def test_balanced_fit(self):
        self.kmeans = KMeans(k=2, max_iterations=100, balanced=True)
        labels, centroids = self.kmeans.fit(self.X)

        cluster_counts = [labels.count(0), labels.count(1)]
        print("\nCluster counts:", cluster_counts)

        self.assertTrue(max(cluster_counts) - min(cluster_counts) < len(self.X) * 0.1)  # Example threshold


if __name__ == "__main__":
    unittest.main()
