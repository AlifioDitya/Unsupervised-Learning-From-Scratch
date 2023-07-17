import numpy as np
from sklearn.metrics import silhouette_score

class KMeans:
    '''
    K-means is a centroid-based clustering algorithm.
    It partitions data into k clusters by iteratively assigning each data point to the cluster
    with the nearest mean and then calculating the new mean of each cluster.

    How it works:
    1. Randomly choose k centroids.
    2. Assign each data point to the closest centroid.
    3. Calculate the new centroids as the mean of the data points assigned to each centroid.
    4. Repeat steps 2 and 3 until the centroids don't change or the max number of iterations is reached.
    '''

    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.inertia = None
        
    def fit(self, data):
        data = np.array(data)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.centroids = {}
        
        # Initialize centroids at random, uniform locations
        for i in range(self.n_clusters):
            self.centroids[i] = data[np.random.randint(0, len(data))]
        
        # Iterate to find the best centroids
        for _ in range(self.max_iter):
            self.classifications = {}
            
            for i in range(self.n_clusters):
                self.classifications[i] = []
                
            # Calculate distances and assign data points to clusters
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
            # Calculate the new centroids and inertia
            new_centroids = []
            inertia = 0
            for classification in self.classifications:
                cluster_points = self.classifications[classification]

                # Calculate the mean of the cluster
                new_centroid = np.average(cluster_points, axis=0)
                new_centroids.append(new_centroid)
                inertia += np.sum((cluster_points - new_centroid) ** 2)
            
            # Update centroids and inertia
            self.centroids = {i: new_centroids[i] for i in range(self.n_clusters)}
            self.inertia = inertia
                
    def predict(self, data):
        # Check if data is a single row or multiple rows
        if len(data.shape) == 1:
            data = np.array([data])

        # Calculate distances and assign data points to clusters
        predictions = []
        for featureset in data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            predictions.append(classification)

        return predictions
    
    def score(self):
        return -self.inertia # Lower inertia is better, sklearn convention.
    
    def silhouette(self, data):
        return silhouette_score(data, self.predict(data))