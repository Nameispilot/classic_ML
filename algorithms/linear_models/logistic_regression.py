import pandas as pd
import numpy as np
import random

class MyLogReg():
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
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) 
    
    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _get_learning_rate(self, iteration):
        """Возвращает learning_rate для текущей итерации"""
        if callable(self.learning_rate):
            return self.learning_rate(iteration)
        return self.learning_rate
    
    def metric_marks(self, y, y_pred) -> list:
        tp =  ((y == 1) & (y_pred == 1)).sum()  
        tn = ((y == 0) & (y_pred == 0)).sum()
        fn = ((y == 1) & (y_pred == 0)).sum()
        fp = ((y == 0) & (y_pred == 1)).sum()
        return [tp, tn, fn, fp]

    def calc_metric(self, metric: str = 'accuracy', y: pd.Series = None, y_pred: pd.Series = None):
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
            positives = np.sum(y == 1)
            negatives = np.sum(y == 0)

            y_pred = np.round(y_pred, 10)

            sorted_idx = np.argsort(-y_pred)
            y_sorted = np.array(y)[sorted_idx]
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
        
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_seed)
        
        X = X.values
        y = y.values
        n_samples = X.shape[0]
        
        # Добавляем единичный столбец для intercept
        X = self._add_intercept(X)
        
        # Инициализируем веса единицами
        weights = np.ones(X.shape[1])
        
        eps = 1e-15
        for i in range(1, self.n_iter + 1):
            current_lr = self._get_learning_rate(i)
            
            # Выбираем мини-пакет
            sample_rows_idx = range(n_samples)
            batch_size = self.sgd_sample if self.sgd_sample else n_samples
            if isinstance(batch_size, float):
                batch_size = int(n_samples * batch_size)
            sample_idx = random.sample(range(n_samples), batch_size)
            X_batch = X[sample_idx]
            y_batch = y[sample_idx]
            
            z = np.dot(X_batch, weights)
            y_pred_batch = self._sigmoid(z)
            
            gradient = np.dot(X_batch.T, (y_pred_batch - y_batch)) / len(y_batch)
            
            y_pred_full = self._sigmoid(np.dot(X, weights))
            log_loss = -np.mean(y * np.log(y_pred_full + eps) + (1 - y) * np.log(1 - y_pred_full + eps))
            if self.reg:
                if self.reg == 'l1' or self.reg == 'elasticnet':
                    gradient += self.l1_coef * np.sign(weights)
                    log_loss += self.l1_coef * np.sum(np.abs(weights))
                if self.reg == 'l2' or self.reg == 'elasticnet':
                    gradient += self.l2_coef * 2 * weights
                    log_loss += self.l2_coef * np.sum(weights ** 2)
            
            weights -= current_lr * gradient
            self.weights = weights
            
            # Вычисление функции потерь и метрики для лога
            if verbose and i % verbose == 0:
                y_pred_proba = self._sigmoid(np.dot(X, self.weights))
                y_pred = (y_pred_proba >= 0.5).astype(int)

                if verbose and (i == 1 or (i % verbose) == 0):
                    print(f'{i}| loss: {log_loss}', end= '')
                if self.metric:
                    metr = self.calc_metric(self.metric, y, y_pred)
                    print(f'| {self.metric}: {metr}')
        y_pred = 1 / (1 + np.exp(-X.dot(weights)))
        metr = self.calc_metric(self.metric, y, y_pred)     
        self.best_metric = metr   
    
    def get_coef(self):
        return self.weights[1:]   
    
    def get_best_score(self):
        return self.best_metric         
                
    def predict_proba(self, X: pd.DataFrame):
        # Проверяем, что модель обучена
        if self.weights is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit().") 
        X = X.values
        X = self._add_intercept(X)
        z = np.dot(X, self.weights)
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)  
        return (proba >= threshold).astype(int)         
    
    
from sklearn.datasets import make_classification
    
X, y = make_classification(n_samples=10, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

logreg = MyLogReg(learning_rate=lambda iter: 0.5 * (0.85 ** iter), n_iter=10, metric='roc_auc', reg='l1')
logreg.fit(X, y, verbose=1)
        

