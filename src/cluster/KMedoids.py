import numpy as np
from sklearn.metrics import silhouette_score

class KMedoids:
    '''
    K-medoids is a medoid-based clustering algorithm.
    It is similar to k-means, except that it chooses datapoints as centers (medoids or exemplars).

    How it works:
    1. Randomly choose k medoids.
    2. Assign each data point to the closest medoid.
    3. Calculate the new medoids as the medoid of the data points assigned to each medoid.
    4. Repeat steps 2 and 3 until the medoids don't change or the max number of iterations is reached.
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

        self.medoids = {}
        
        # Initialize medoids at random, uniform locations
        for i in range(self.n_clusters):
            self.medoids[i] = data[np.random.randint(0, len(data))]
        
        # Iterate to find the best medoids
        for _ in range(self.max_iter):
            self.classifications = {}
            
            for i in range(self.n_clusters):
                self.classifications[i] = []
                
            # Calculate distances and assign data points to clusters
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.medoids[medoid]) for medoid in self.medoids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
            # Calculate the new medoids and inertia
            new_medoids = []
            inertia = 0
            for classification in self.classifications:
                cluster_points = np.array(self.classifications[classification])

                # Find the point in the cluster that minimizes the sum of distances to all other points in the cluster
                new_medoid = cluster_points[np.argmin(np.sum(np.abs(cluster_points[:, None] - cluster_points), axis=-1))]
                new_medoids.append(new_medoid)
                inertia += np.sum((cluster_points - new_medoid) ** 2)
            
            # Update medoids and inertia
            self.medoids = {i: new_medoids[i] for i in range(self.n_clusters)}
            self.inertia = inertia
    
    def predict(self, data):
        # Check if data is a single row or multiple rows
        if len(data.shape) == 1:
            data = np.array([data])

        # Calculate distances and assign data points to clusters
        predictions = []
        for featureset in data:
            distances = [np.linalg.norm(featureset - self.medoids[medoid]) for medoid in self.medoids]
            classification = distances.index(min(distances))
            predictions.append(classification)

        return predictions
    
    def score(self):
        return -self.inertia
    
    def silhouette(self, data):
        return silhouette_score(data, self.predict(data))