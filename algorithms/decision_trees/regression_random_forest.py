import pandas as pd
import numpy as np
import random
from regression_decision_tree import TreeReg

class MyForestReg:
    def __init__(self, n_estimators:int=10, max_features:float=0.5, max_samples:float=0.5, random_state:int=42, \
                max_depth:int=5, min_samples_split:int=2, max_leafs:int=20, bins=16, oob_score=None):
        self.n_estimators = n_estimators
        self.max_features = max_features 
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.oob_score = oob_score
        self.oob_score_ = None
        self.oob_predictions = {}  # Словарь для хранения OOB-предсказаний каждого дерева
        self.oob_counts = {}       # Счетчик, сколько раз каждый образец был OOB

        self.forest_structure = []
        self.leafs_cnt = 0
        self.fi = {}

    def __str__(self):
        return f"MyForestReg class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples},\
            max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}, random_state={self.random_state}" 

    def _mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def _mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def _rmse(self, y_true, y_pred):
        return np.sqrt(self._mse(y_true, y_pred))

    def _mape(self, y_true, y_pred):
        return 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    
    def _r2(self, y_true, y_pred):
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        random.seed(self.random_state)
        init_cols = list(X.columns)
        init_rows_cnt = X.shape[0]
        cols_smpl_cnt = round(X.shape[1] * self.max_features)
        rows_smpl_cnt = round(X.shape[0] * self.max_samples)

        self.fi = {col: 0 for col in X.columns}
        N = X.shape[0]
        self.oob_predictions = {i: 0.0 for i in range(init_rows_cnt)}
        self.oob_counts = {i: 0 for i in range(init_rows_cnt)}

        for _ in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)
            oob_rows = [i for i in range(init_rows_cnt) if i not in rows_idx]

            tree = TreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            tree.fit(X.loc[rows_idx, cols_idx], y[rows_idx], N)

            self.forest_structure.append(tree)
            self.fi = {key: tree.fi.get(key, 0) + self.fi.get(key, 0) for key in set(tree.fi) | set(self.fi)}
            self.leafs_cnt += tree.leafs_cnt   

            # Делаем предсказание для OOB-образцов текущего дерева
            if oob_rows:
                X_oob = X.iloc[oob_rows][cols_idx]
                y_oob_pred = tree.predict(X_oob)
                
                for i, idx in enumerate(oob_rows):
                    if idx in self.oob_predictions:
                        self.oob_predictions[idx] += y_oob_pred[i]
                        self.oob_counts[idx] += 1

        # Усредняем OOB-предсказания и вычисляем метрику
        if self.oob_score:
            valid_indices = [i for i in self.oob_counts if self.oob_counts[i] > 0]
            if not valid_indices:
                self.oob_score_ = None
                return

            y_true = y.iloc[valid_indices]
            y_pred = np.array([self.oob_predictions[i] / self.oob_counts[i] for i in valid_indices])

            # Проверка на NaN/Inf (дополнительная защита)
            mask = np.isfinite(y_pred) & np.isfinite(y_true)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if len(y_true) == 0:
                self.oob_score_ = None
                return
            
            self.oob_score_ = getattr(self, f'_{self.oob_score}')(y_true, y_pred)              

    def predict(self, X: pd.DataFrame):
        return np.array([tree.predict(X) for tree in self.forest_structure]).mean(axis=0)   
     
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

forest = MyForestReg(n_estimators=10, max_depth=5, max_features=0.5, max_samples=2, max_leafs=15, bins=15, oob_score='r2')            
forest.fit(X, y)
forest.predict(X)