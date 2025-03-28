import pandas as pd
import numpy as np
import random
from regression_decision_tree import TreeReg

class MyForestReg:
    def __init__(self, n_estimators:int=10, max_features:float=0.5, max_samples:float=0.5, random_state:int=42, \
                max_depth:int=5, min_samples_split:int=2, max_leafs:int=20, bins=16):
        self.n_estimators = n_estimators
        self.max_features = max_features 
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.forest_structure = []
        self.leafs_cnt = 0
        self.fi = {}

    def __str__(self):
        return f"MyForestReg class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples},\
            max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}, random_state={self.random_state}" 

    def fit(self, X: pd.DataFrame, y: pd.Series):
        random.seed(self.random_state)
        init_cols = list(X.columns)
        init_rows_cnt = X.shape[0]
        cols_smpl_cnt = round(X.shape[1] * self.max_features)
        rows_smpl_cnt = round(X.shape[0] * self.max_samples)

        self.fi = {col: 0 for col in X.columns}
        N = X.shape[0]

        for _ in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)

            tree = TreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            tree.fit(X.loc[rows_idx, cols_idx], y[rows_idx], N)

            self.forest_structure.append(tree)
            
            self.fi = {key: tree.fi.get(key, 0) + self.fi.get(key, 0) for key in set(tree.fi) | set(self.fi)}

            self.leafs_cnt += tree.leafs_cnt   

    def predict(self, X: pd.DataFrame):
        return np.array([tree.predict(X) for tree in self.forest_structure]).mean(axis=0)    

from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

forest = MyForestReg(n_estimators=6, max_depth=2, max_features=0.6, max_samples=0.5)            
forest.fit(X, y)
forest.predict(X)