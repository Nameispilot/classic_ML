import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import random
import copy
from classification_decision_tree import MyTreeClf
from linear_models.logistic_regression import MyLogReg
from metric_algorithms.knn_classification import MyKNNClf
from collections import defaultdict

class MyBaggingClf:
    def __init__(self, estimator=None, n_estimators:int=10, max_samples:float=1.0, random_state:int=42, oob_score=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score = oob_score

        self.estimators = []
        self.oob_score_ = None
    
    def __str__(self):
         return f"MyBaggingClf class: estimator={self.estimator}, n_estimators={self.n_estimators}, max_samples={self.max_samples}, random_state={self.random_state}"    

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
        return self     