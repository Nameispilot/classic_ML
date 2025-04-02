import pandas as pd
import numpy as np
import random

class MySVM():
    def __init__(self, n_iter: int=10, learning_rate: float=0.001, weights=None, b=None, C: int =1, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.b = b
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state
    
    def __str__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def get_coef(self):
        return (self.weights, self.b)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)
        labels = np.where(y == 0, -1, 1)

        num_observations, num_features = X.shape
        weights = np.ones(num_features)
        b = 1


        """Подбор batch_size"""
        if self.sgd_sample:
            if isinstance(self.sgd_sample, float):
                batch_size = int(X.shape[0] * self.sgd_sample)
            elif isinstance(self.sgd_sample, int):
                batch_size = self.sgd_sample
            num_observations = batch_size 
        grad_w, grad_b = 0.0, 0.0
        for i in range(1, self.n_iter+1):
            sample_rows_idx = random.sample(range(X.shape[0]), batch_size)
            if self.sgd_sample == None:
                X_batch = X
            else:    
                X_batch = X.iloc[sample_rows_idx]  
            for j, row in X_batch.iterrows():
                if labels[j] * (np.dot(weights, row) + b) >= 1:
                    grad_w = 2 * weights
                    grad_b = 0.0
                else:
                    grad_w = 2*weights - self.C * labels[j] * row
                    grad_b = - self.C * labels[j]
                weights = weights - self.learning_rate * grad_w
                b = b - self.learning_rate * grad_b   

            self.weights = weights
            self.b = b

            # Вычисляем отступы
            margins = y * (np.dot(X, self.weights) + self.b)
            # Вычисляем hinge loss: max(0, 1 - margins)
            hinge_loss = np.maximum(0, 1 - margins)
            # Вычисляем регуляризационный член: ||w||^2
            regularization = np.dot(self.weights, self.weights)
            # Вычисляем общий loss
            loss = regularization + (1 / num_observations) * self.C * np.sum(hinge_loss)

            if verbose and (i == 1 or (i % verbose) == 0):
                print(f'{i}| loss: {loss}')

    def predict(self, X: pd.DataFrame):
        y = np.sign(np.dot(X, self.weights) + self.b)
        return np.where(y == -1, 0, 1)            

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

svm = MySVM()
svm.fit(X, y, verbose=2)
svm.predict(X)