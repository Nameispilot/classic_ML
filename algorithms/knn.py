import pandas as pd
import numpy as np
import scipy.spatial

class KNNClf:
    def __init__(self, k: int = 3, metric: str = 'euclidian'):
        self.k = k
        self.train_size = 0
        self.X = None
        self.y = None
        self.metric = metric

    def __str__(self):
        return f'KNNClf class: k={self.k}'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = X.shape

    def euclidean_dist(self, row: pd.Series):
        distance = np.sum((self.X.values - row.values) ** 2, axis=1) ** 0.5  
        sorted_indices = np.argsort(distance)[:self.k]
        return self.y.iloc[sorted_indices].mean()
    
    def chebyshev_dist(self, row: pd.Series):
        distance = np.max(np.abs(self.X.values - row.values), axis=1)
        sorted_indices = np.argsort(distance)[:self.k]
        return self.y.iloc[sorted_indices].mean()
    
    def manhattan_dist(self, row: pd.Series):
        distance = np.sum(np.absolute(self.X.values - row.values), axis=1)
        sorted_indices = np.argsort(distance)[:self.k]
        return self.y.iloc[sorted_indices].mean()
    
    def cosine_dist(self, row: pd.Series):
        cosine_similarities = np.dot(self.X.values, row.values) / (np.linalg.norm(self.X.values, axis=1) * np.linalg.norm(row.values)) 
        distance = 1 - cosine_similarities
        sorted_indices = np.argsort(distance)[:self.k]
        return self.y.iloc[sorted_indices].mean()

    def predict(self, test: pd.DataFrame):
        logits = self.predict_proba(test)
        return (logits > 0.5).astype(int)
          
    def predict_proba(self, test: pd.DataFrame):
        return test.apply(getattr(self, f'{self.metric}_dist'), axis=1)
        
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=5, n_informative=2, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

X_test, _ = make_classification(n_samples=100, n_features=5, n_informative=2, random_state=42)
X_test = pd.DataFrame(X)
X_test.columns = [f'col_{col}' for col in X_test.columns]

knn_clf = KNNClf(5, 'cosine')
knn_clf.fit(X, y)
print(knn_clf.predict(X_test))   
print(knn_clf.predict_proba(X_test))  
    