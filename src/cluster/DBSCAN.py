import numpy as np
from sklearn.metrics import silhouette_score

class DBSCAN:
    '''
    DBSCAN is a density-based clustering algorithm. 
    It groups together points that are close to each other based on a distance measurement 
    (usually Euclidean distance) and a minimum number of points.

    How it works:
    1. Randomly choose a point that has not been visited.
    2. Find all points in the neighborhood of that point based on some distance measurement.
    3. If there are at least min_samples points within distance eps of the current point,
        add all of those points to the current cluster. The current point is now considered visited.
    4. If there are not min_samples points within distance eps of the current point,
        label the current point as noise. This is the case where a cluster is completed.
    5. Repeat steps 1 to 4 until all points are visited. If a point was labeled noise,
        but it is in the neighborhood of a cluster, assign the point to that cluster.
    '''
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        X = np.array(X)

        # Init labels
        self.labels = np.zeros(len(X))
        cluster_id = 1

        # Iterate through all points
        for i in range(len(X)):
            # If point is already assigned, skip
            if self.labels[i] != 0:
                continue

            # Find all neighbors
            neighbors = self.find_neighbors(X, i)

            # If not enough neighbors, mark as noise
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
                continue

            # Assign all neighbors to cluster
            self.labels[i] = cluster_id
            for j in neighbors:
                self.labels[j] = cluster_id

            # Increment cluster id
            cluster_id += 1

        return self.labels
    
    def find_neighbors(self, X, i):
        neighbors = []
        for j in range(len(X)):
            if i == j:
                continue
            if self.dist(X[i], X[j]) < self.eps:
                neighbors.append(j)
        return neighbors
    
    def dist(self, x1, x2):
        # Euclidean distance
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def get_labels(self):
        return self.labels
    
    def silhouette(self, X):
        X = np.array(X)
        return silhouette_score(X, self.labels)