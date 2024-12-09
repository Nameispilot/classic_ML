import pandas as pd
import numpy as np

class LogReg():
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.1, metric: str = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.best_metric = 0.0

    def __str__(self):
        return f'LogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'    
    
    def loss(self, loss: str = 'LogLoss', y: pd.Series = None, y_pred : pd.Series = None, n: int = 1) -> float  :
        if loss == 'LogLoss':
            return -1.0 / n * (y.dot(np.log(y_pred + 1e-5)) + (1-y).dot(np.log(1-y_pred + 1e-5))).sum()
        
    def gradient(self, loss: str = 'LogLoss', X: pd.DataFrame = None, y: pd.Series = None, y_pred: pd.Series = None) -> float:
        if loss == 'LogLoss':
            return (y_pred - y).dot(X) / X.shape[0] 

    def metric_marks(self, y, y_pred) -> list:
        tp =  ((y == 1) & (y_pred == 1)).sum()  
        tn = ((y == 0) & (y_pred == 0)).sum()
        fn = ((y == 1) & (y_pred == 0)).sum()
        fp = ((y == 0) & (y_pred == 1)).sum()
        return [tp, tn, fn, fp]

    def calc_metric(self, y: pd.Series = None, y_pred: pd.Series = None):
        metric = self.metric
        labels = (y_pred > 0.5).astype(int)
        tp, tn, fn, fp  = self.metric_marks(y, labels) # returns as following: [tp, tn, fn, fp] 
        if metric == 'accuracy':
            return (tp + tn) / (tp + tn + fp + fn)
        elif metric == 'precision':
            return tp / (tp + fp)
        elif metric == 'recall':
            return tp / (tp + fn)
        elif metric == 'f1':
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * precision * recall / (precision + recall)
        elif metric == 'roc_auc':
            y_pred = y_pred.round(10)
            df = pd.concat([y_pred, y], axis=1)
            df = df.sort_values(by=0, ascending=False)  

            positives = df[df[1] == 1]
            negatives = df[df[1] == 0]
            total = 0
            for current_score in negatives[0]:
                score_higher = (positives[0] > current_score).sum()
                score_equal = (positives[0] == current_score).sum()
                total += score_higher + 0.5 * score_equal
            return total / (positives.shape[0] * negatives.shape[0])
        
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: object = False):
        X.insert(0, 'w0', 1.0)
        num_observations, num_features = X.shape
        weights = np.ones(num_features)

        for i in range(1, self.n_iter+1):
            y_pred = 1 / (1 + np.exp(-X.dot(weights)))

            loss = self.loss('LogLoss', y, y_pred, num_observations)
            grad = self.gradient('LogLoss', X, y, y_pred)
            weights = weights - self.learning_rate*grad
            self.weights = weights

            if verbose and (i == 1 or (i % verbose) == 0):
                print(f'{i}| loss: {loss}', end= '')
                if self.metric:
                    metr = self.calc_metric(self.metric, y, y_pred)
                    print(f'| {self.metric}: {metr}')
        y_pred = 1 / (1 + np.exp(-X.dot(weights)))
        metr = self.calc_metric(self.metric, y, y_pred)     
        self.best_metric = metr   

    
    def get_coef(self) -> np.array:
        """Вектор весов без первого параметра"""
        return self.weights[1:]   

    def predict_proba(self, X: pd.DataFrame = None) -> pd.Series: 
        X = X.copy()
        X.insert(0, 'w0', 1.0)
        return  1 / (1 + np.exp(-X.dot(self.weights)))
    
    def predict(self, X: pd.DataFrame = None) -> pd.Series:
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)  

    def get_best_score(self):
        return self.best_metric  

"""Данные для тестов"""
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]           

reg = LogReg(n_iter=130, learning_rate=0.03, metric='roc_auc')
reg.fit(X, y, verbose=False)
print(reg.get_best_score())


        

