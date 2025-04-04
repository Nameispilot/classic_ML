import pandas as pd
import numpy as np
import random
from classification_decision_tree import MyTreeClf

class MyForestClf:
    def __init__(self, n_estimators:int=10, max_features:float=0.5, max_samples:float=0.5, random_state:int=42, \
                max_depth:int=5, min_samples_split:int=2, max_leafs:int=20, bins=16, criterion='entropy', oob_score=None):
        self.n_estimators = n_estimators
        self.max_features = max_features 
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion
        self.oob_score = oob_score
        self.oob_score_ = None
        self.oob_predictions = {}  # Словарь для хранения OOB-предсказаний каждого дерева
        self.oob_counts = {}       # Счетчик, сколько раз каждый образец был OOB

        self.forest_structure = []
        self.leafs_cnt = 0
        self.fi = {}

    def __str__(self):
        return f"MyForestClf class: n_estimators={self.n_estimators}, max_features={self.max_features}, \
            max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, \
                max_leafs={self.max_leafs}, bins={self.bins}, criterion={self.criterion}, random_state={self.random_state}"

    def _precision(self, y_true, y_pred):
        y_pred = (y_pred > 0.5).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return tp / (tp + fp)
    
    def _recall(self, y_true, y_pred):
        y_pred = (y_pred > 0.5).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        return tp / (tp + fn)
        
    def _accuracy(self, y_true, y_pred):
        y_pred = (y_pred > 0.5).astype(int)
        return (y_true == y_pred).mean()

    def _f1(self, y_true, y_pred):
        y_pred = (y_pred > 0.5).astype(int)
        precision = self._precision(y_true, y_pred)
        recall = self._recall(y_true, y_pred)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def _roc_auc(self, y_true, y_pred):
        positives = np.sum(y_true == 1)
        negatives = np.sum(y_true == 0)

        y_pred = np.round(y_pred, 10)

        sorted_idx = np.argsort(-y_pred)
        y_sorted = np.array(y_true)[sorted_idx]
        y_prob_sorted = y_pred[sorted_idx]

        roc_auc_score = 0

        for prob, pred in zip(y_prob_sorted, y_sorted):
            if pred == 0:
                roc_auc_score += (
                    np.sum(y_sorted[y_prob_sorted > prob])
                    + np.sum(y_sorted[y_prob_sorted == prob]) / 2
                )

        roc_auc_score /= positives * negatives

        return roc_auc_score

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

            tree = MyTreeClf(self.max_depth, self.min_samples_split, self.max_leafs, self.bins, N=N)
            tree.fit(X.loc[rows_idx, cols_idx], y[rows_idx])

            self.forest_structure.append(tree)
            self.fi = {key: tree.fi.get(key, 0) + self.fi.get(key, 0) for key in set(tree.fi) | set(self.fi)}
            self.leafs_cnt += tree.leafs_cnt 

            # Делаем предсказание для OOB-образцов текущего дерева
            if oob_rows:
                X_oob = X.iloc[oob_rows][cols_idx]
                y_oob_pred = tree.predict_proba(X_oob)
                
                for i, idx in enumerate(oob_rows):
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

    def predict(self, X: pd.DataFrame, type: str):
        if type == 'mean':
            proba = self.predict_proba(X)
            return (proba > 0.5).astype(int)
        elif type == 'vote':
            all_predictions = np.array([tree.predict(X) for tree in self.forest_structure]).T
            result = []
            for row in all_predictions:
                values, counts = np.unique(row, return_counts=True)
                max_count = np.max(counts)
                candidates = values[counts == max_count]
                result.append(np.min(candidates))
            return np.array(result)    
        else:
            raise ValueError("Неправильно выбранный type!")

    def predict_proba(self, X: pd.DataFrame):
        return np.array([tree.predict_proba(X) for tree in self.forest_structure]).mean(axis=0)


from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

forest = MyForestClf(n_estimators=6, max_depth=2, max_features=0.6, max_samples=0.5, oob_score='accuracy')            
forest.fit(X, y)
print(forest.predict(X, type='vote'))

