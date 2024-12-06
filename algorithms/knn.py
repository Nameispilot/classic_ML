import pandas as pd
import numpy as np

class KNNClf:
    def __init__(self, k: int = 3):
        self.k = k
        self.train_size = 0
        self.X = None
        self.y = None

    def __str__(self):
        return f'KNNClf class: k={self.k}'
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = X.shape

    def euclidean_class(self, row: pd.Series):
        distance = np.sum((self.X.values - row.values) ** 2, axis=1) ** 0.5  
        sorted_indices = np.argsort(distance)[:self.k]
        if self.y.iloc[sorted_indices].mean() >= 0.5:
            return 1
        else:
            return 0

    def euclidean_proba(self, row: pd.Series):
        distance = np.sum((self.X.values - row.values) ** 2, axis=1) ** 0.5     
        sorted_indices = np.argsort(distance)[:self.k]
        return self.y.iloc[sorted_indices].mean()

    def predict(self, test: pd.DataFrame):
        return test.apply(self.euclidean_class, axis=1)
        
    def predict_proba(self, test: pd.DataFrame):
        return test.apply(self.euclidean_proba, axis=1)

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=5, n_informative=2, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

X_train, y_train = make_classification(n_samples=100, n_features=5, n_informative=2, random_state=42)
X_train = pd.DataFrame(X)
X_train.columns = [f'col_{col}' for col in X_train.columns]

knn_clf = KNNClf(5)
knn_clf.fit(X, y)
print(knn_clf.predict(X_train))   
print(knn_clf.predict_proba(X_train))  
    