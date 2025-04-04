import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import random
import copy
from regression_decision_tree import MyTreeReg
from linear_models.linear_regression import MyLineReg
from metric_algorithms.knn_regression import MyKNNReg
from collections import defaultdict

class MyBaggingReg:
    def __init__(self, estimator=None, n_estimators:int=10, max_samples:float=1.0, random_state:int=42, oob_score=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score = oob_score

        self.estimators = []
        self.oob_score_ = None

    def __str__(self):
         return f"MyBaggingReg class: estimator={self.estimator}, n_estimators={self.n_estimators}, max_samples={self.max_samples}, random_state={self.random_state}"     
        
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

        init_rows_cnt = X.shape[0]
        rows_num_list = list(range(init_rows_cnt))
        rows_smpl_cnt = round(X.shape[0] * self.max_samples)
        
        self.oob_predictions = defaultdict(list)

        all_sample_rows_idx = [
            random.choices(rows_num_list, k=rows_smpl_cnt)
            for _ in range(self.n_estimators)
        ]

        for sample_rows_idx in all_sample_rows_idx:
            X_sample = X.iloc[sample_rows_idx]
            y_sample = y.iloc[sample_rows_idx]
            oob_rows = [i for i in range(init_rows_cnt) if i not in sample_rows_idx]

            model = copy.deepcopy(self.estimator)
            model.fit(X_sample, y_sample)
            self.estimators.append(model)
            
            if self.oob_score and oob_rows:
                X_oob_sample = X.iloc[oob_rows]
                y_oob_pred = model.predict(X_oob_sample)
                for idx, pred in zip(oob_rows, y_oob_pred):
                    self.oob_predictions[idx].append(pred)
                    
        if self.oob_score:  
            oob_y_true = []
            oob_y_pred = []
            
            for idx, preds in self.oob_predictions.items():
                if preds:
                    oob_y_true.append(y.iloc[idx])
                    oob_y_pred.append(np.mean(preds))
            if oob_y_true:
                oob_y_true = np.array(oob_y_true)
                oob_y_pred = np.array(oob_y_pred)  
                self.oob_score_ = getattr(self, f'_{self.oob_score}')(oob_y_true, oob_y_pred)              
        return self 
    
    def predict(self, X: pd.DataFrame):
        if not self.estimators:
            raise ValueError("Модель не обучена. Сначала вызовите fit()")
        predictions = []
        for model in self.estimators:
            pred = model.predict(X)
            predictions.append(pred)
        # Транспонируем, чтобы перейти от "модели → примеры" к "примеры → модели"
        predictions_per_sample = list(zip(*predictions))    
        
        # Усредняем предсказания для каждого примера
        final_predictions = [sum(preds)/len(preds) for preds in predictions_per_sample]
        
        return np.array(final_predictions)

from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

bag = MyBaggingReg(estimator=MyLineReg(), oob_score='r2')
bag.fit(X, y)
            