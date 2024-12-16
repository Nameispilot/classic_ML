import pandas as pd
import numpy as np
import random

class LogReg():
    def __init__(self, n_iter: int = 100,
               learning_rate: object = 0.1,
               metric: str = None,
               reg: str = None,
               l1_coef: float = 0.0,
               l2_coef: float = 0.0,
               sgd_sample: object = None,
               random_seed: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_seed = random_seed
        self.best_metric = 0.0

    def __str__(self):
        return f'LogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'    
    
    def loss(self, loss: str = 'LogLoss', y: pd.Series = None, y_pred : pd.Series = None, n: int = 1) -> float  :
        if loss == 'LogLoss':
            return -1.0 / n * (y.dot(np.log(y_pred + 1e-5)) + (1-y).dot(np.log(1-y_pred + 1e-5))).sum()
        
    def gradient(self, loss: str = 'LogLoss', X: pd.DataFrame = None, y: pd.Series = None, y_pred: pd.Series = None) -> float:
        if loss == 'LogLoss':
            return (y_pred - y).dot(X) / X.shape[0] 
    
    def precision(self, y: pd.Series = None, y_pred: pd.Series = None) -> float:
        y_pred = (y_pred > 0.5).astype(int)
        tp = ((y == 1) & (y_pred == 1)).sum()
        fp = ((y == 0) & (y_pred == 1)).sum()
        return tp / (tp + fp)
    
    def recall(self, y: pd.Series = None, y_pred: pd.Series = None) -> float:
        y_pred = (y_pred > 0.5).astype(int)
        tp = ((y == 1) & (y_pred == 1)).sum()
        fn = ((y == 1) & (y_pred == 0)).sum()
        return tp / (tp + fn)
        
    def accuracy(self, y: pd.Series = None, y_pred: pd.Series = None) -> float:
        y_pred = (y_pred > 0.5).astype(int)
        return (y == y_pred).mean()

    def roc_auc(self, y: pd.Series = None, y_pred: pd.Series = None) -> float:
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
    
    def calc_reg(self, weights) -> float:
        """Подсчет регуляризации"""
        if self.reg == 'l1':
            reg_loss = self.l1_coef * np.sum(np.absolute(weights))
            reg_grad = self.l1_coef * np.sign(weights)
        elif self.reg == 'l2':
            reg_loss = self.l2_coef * np.sum(np.power(weights, 2))
            reg_grad = self.l2_coef * 2 * weights
        elif self.reg == 'elasticnet':
            reg_loss = self.l1_coef * np.sum(np.absolute(weights)) + self.l2_coef * np.sum(np.power(weights, 2))
            reg_grad = self.l1_coef * np.sign(weights) + self.l2_coef * 2 * weights
        return reg_loss, reg_grad

    """Обучение"""
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: object = False):
        random.seed(self.random_seed)
        X.insert(0, 'w0', 1.0)
        num_observations, num_features = X.shape
        weights = np.ones(num_features)

        """Подбор batch_size"""
        if self.sgd_sample:
            if isinstance(self.sgd_sample, float):
                batch_size = int(X.shape[0] * self.sgd_sample)
            elif isinstance(self.sgd_sample, int):
                batch_size = self.sgd_sample
            num_observations = batch_size
        else:
            batch_size = num_observations

        """Обучение"""
        for i in range(1, self.n_iter+1):
            sample_rows_idx = random.sample(range(X.shape[0]), batch_size)
            X_batch = X.iloc[sample_rows_idx]
            y_batch = y.iloc[sample_rows_idx]
            y_pred = 1 / (1 + np.exp(-X.dot(weights)))
            
            y_pred_batch = 1 / (1 + np.exp(-X_batch.dot(weights)))

            """Регуляризация"""
            reg_loss, reg_grad = 0.0, 0.0
            if self.reg:
                reg_loss, reg_grad = self.calc_reg(weights)

            """Проверка скорости обучения"""
            if callable(self.learning_rate):
                lr = self.learning_rate(i)
            else:
                lr = self.learning_rate    

            """Расчет градиента и обновление весов"""
            loss = self.loss('LogLoss', y, y_pred, num_observations) + reg_loss
            grad = self.gradient('LogLoss', X_batch, y_batch, y_pred_batch) + reg_grad
            weights = weights - lr*grad
            self.weights = weights

            """Вывод логов"""
            if verbose and (i == 1 or (i % verbose) == 0):
                print(f'{i}| loss: {loss}', end= '')
                if self.metric:
                    metr = getattr(self, f'{self.metric}')(y, y_pred)
                    print(f'| {self.metric}: {metr}')
        if self.metric:
            y_pred = 1 / (1 + np.exp(-X.dot(weights)))
            self.best_metric = getattr(self, f'{self.metric}')(y, y_pred)     
    
    def get_coef(self) -> np.array:
        """Вектор весов без первого параметра"""
        return self.weights[1:]   

    def predict_proba(self, X: pd.DataFrame = None) -> pd.Series: 
        if X.shape[1] != self.weights.shape[0]:
            X.insert(0, 'w0', 1.0)
        return  1 / (1 + np.exp(-X.dot(self.weights)))
    
    def predict(self, X: pd.DataFrame = None) -> pd.Series:
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)  

    def get_best_score(self):
        return self.best_metric  

"""Данные для тестов"""
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=300, n_features=5, n_informative=2, random_state=42)
X_train = pd.DataFrame(X[:200])
y_train = pd.Series(y[:200])
X_train.columns = [f'col_{col}' for col in X_train.columns]

X_test = pd.DataFrame(X[201:])
X_test.columns = [f'col_{col}' for col in X_test.columns]   

reg = LogReg(n_iter=100, learning_rate=lambda iter: 0.5 * (0.85 ** iter), metric='recall', sgd_sample=0.1)
reg.fit(X_train, y_train, verbose=10)
print(reg.predict(X_test).sum())


        

