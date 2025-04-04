import pandas as pd
import numpy as np

class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        self.k = k
        self.train_size = 0
        self.X = None
        self.y = None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        return f'KNNClf class: k={self.k}'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = X.shape

    def euclidean_dist(self, row: pd.Series):
        distance = np.sum((self.X.values - row.values) ** 2, axis=1) ** 0.5  
        return distance
    
    def chebyshev_dist(self, row: pd.Series):
        distance = np.max(np.abs(self.X.values - row.values), axis=1)
        return distance
    
    def manhattan_dist(self, row: pd.Series):
        distance = np.sum(np.absolute(self.X.values - row.values), axis=1)
        return distance
    
    def cosine_dist(self, row: pd.Series):
        cosine_similarities = np.dot(self.X.values, row.values) / (np.linalg.norm(self.X.values, axis=1) * np.linalg.norm(row.values)) 
        distance = 1 - cosine_similarities
        return distance
        
    def _calc_row(self, row: pd.Series):
        distances = getattr(self, f'{self.metric}_dist')(row)
        sorted_indices = np.argsort(distances)[:self.k]
        neighbors = self.y.iloc[sorted_indices].values
        if self.weight == 'uniform':
            weights = np.ones(self.k)
        elif self.weight == 'distance':
            weights = 1 / (distances[sorted_indices])
        elif self.weight == 'rank':
            weights = 1 / (np.arange(self.k) + 1)

        class_weights = np.bincount(neighbors, weights=weights, minlength=2)
        return class_weights[1] / np.sum(class_weights)    

    def predict(self, test: pd.DataFrame):
        return test.apply(lambda row: (self._calc_row(row) > 0.5).astype(int), axis=1)
        
    def predict_proba(self, test: pd.DataFrame):
        return test.apply(self._calc_row, axis=1)