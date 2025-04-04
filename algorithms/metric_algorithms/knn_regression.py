import pandas as pd
import numpy as np

class MyKNNReg():
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        self.k = k
        self.train_size = 0
        self.X = None
        self.y = None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        return f'MyKNNReg class: k={self.k}'   

    def euclidean_dist(self, row: pd.Series):
        distance = np.sum((self.X.values - row.values) ** 2, axis=1) ** 0.5  
        return distance

    def chebyshev_dist(self, row: pd.Series):
        distance = np.max(np.abs(self.X.values - row.values), axis=1)
        return distance
    
    def manhattan_dist(self, row: pd.Series):
        distance = np.sum(np.abs(self.X.values - row.values), axis=1)
        return distance
    
    def cosine_dist(self, row: pd.Series):
        cosine_similarities = np.dot(self.X.values, row.values) / (np.linalg.norm(self.X.values, axis=1) * np.linalg.norm(row.values)) 
        distance = 1 - cosine_similarities
        return distance

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = X.shape   

    def _calc_row(self, row: pd.Series):
        distances = getattr(self, f'{self.metric}_dist')(row)
        sorted_indices = np.argsort(distances)[:self.k]
        neighbors = self.y.iloc[sorted_indices].values

        if self.weight == 'uniform':
            return neighbors.mean()
        elif self.weight == 'rank':
            ri = 1/ (np.arange(self.k) + 1)
            weights = ri / ri.sum()
        elif self.weight == 'distance':
            di = (1 / (distances[sorted_indices]))
            weights = di / di.sum()   
            
        return neighbors.dot(weights).sum()

    def predict(self, test: pd.DataFrame):
        return test.apply(lambda row: self._calc_row(row), axis=1)

'''from sklearn.datasets import make_regression
X, y = make_regression(n_samples=300, n_features=5, n_informative=2, random_state=42)
X_train = pd.DataFrame(X[:200])
y_train = pd.Series(y[:200])
X_train.columns = [f'col_{col}' for col in X_train.columns]

X_test = pd.DataFrame(X[201:])
X_test.columns = [f'col_{col}' for col in X_test.columns]

knn_clf = MyKNNReg(k=5, metric='chebyshev')
knn_clf.fit(X_train, y_train)
print(knn_clf.predict(X_test).sum()) '''



