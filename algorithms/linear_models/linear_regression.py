import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_regression

class LineReg():
  """
    Линейная регрессия

    n_iter : int
        Количество шагов градиентного спуска
    learning_rate : float/сallable
        Скорость обучения градиентного спуска
        Если на вход пришла lambda-функция, то learning_rate вычисляется на каждом шаге на основе lambda-функции
    weights : np.ndarray
        Веса модели, by default None
    metric : str
        Метрика для оценки точности модели
        Принимает одно из следующих значений: mae, mse, rmse, mape, r2
    reg : str
        Вид регуляризации. Принимает одно из следующих значений: l1, l2, elasticnet
    l1_coef : float
        Коэффициент L1 регуляризации. Принимает значения от 0.0 до 1.0
    l2_coef : float
        Коэффициент L2 регуляризации. Принимает значения от 0.0 до 1.0
    sgd_sample : int/float
        Размер батча для стохастического градиентного спуска. Если не задан - используется обычный градиентный спуск
        Может принимать целые числа, либо дробные от 0.0 до 1.0
    random_state : int
        Сид для рандомного подбора примеров
    """
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
    return f'LineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, metric={self.metric}, regularization={self.reg}'

  def calc_metric(self, y, y_pred, num_observations) -> float:
    """Подсчет метрики для оценки точности модели"""
    if self.metric == 'mae':
      return ((y - y_pred).abs().sum()) / num_observations
    elif self.metric == 'mse':
      return ((y_pred - y)**2).sum() / num_observations
    elif self.metric == 'rmse':
      return np.sqrt(((y_pred - y)**2).sum() / num_observations)
    elif self.metric == 'mape':
      return 100 / num_observations * ((y - y_pred)/y).abs().sum()
    elif self.metric == 'r2':
      return 1.0 - ((y - y_pred)**2).sum() / ((y - y.mean())**2).sum()

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

  def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
    """
        Обучение линейной регрессии

        X : pd.DataFrame
            Входные данные вида [num_observations, num_features]
        y : pd.Series
            Целевая переменная
        verbose : int
            Указывает через сколько итераций градиентного спуска будет выводиться в логе, если True
    """
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
      y_pred = X.dot(weights)
      y_pred_batch = X_batch.dot(weights)

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
      mse = ((y_pred - y)**2).sum() / num_observations + reg_loss
      grad = 2 * (y_pred_batch - y_batch).dot(X_batch) / num_observations + reg_grad
      weights = weights - lr*grad
      self.weights = weights

      """Вывод логов"""
      if verbose and (i == 1 or (i % verbose) == 0):
        print(f'{i}| loss: {mse}| learning rate: {lr}', end = '')
        if self.metric:
          print(f'| {self.metric}: {self.calc_metric(y, y_pred, num_observations)}')
        print('\n')

    y_pred = X.dot(weights)
    self.best_metric = self.calc_metric(y, y_pred, num_observations)

  def predict(self, X: pd.DataFrame):
    """Выдача предсказания после обучение"""
    if X.shape[1] != self.weights.shape[0]:
            X.insert(0, 'w0', 1.0)
    return X.dot(self.weights)

  def get_coef(self) -> np.array:
    """Вектор весов без первого параметра"""
    return self.weights[1:]

  def get_best_score(self):
    """Метрика после обучения"""
    return self.best_metric

"""Данные для тестов"""
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=300, n_features=5, n_informative=2, random_state=42)
X_train = pd.DataFrame(X[:200])
y_train = pd.Series(y[:200])
X_train.columns = [f'col_{col}' for col in X_train.columns]

X_test = pd.DataFrame(X[201:])
X_test.columns = [f'col_{col}' for col in X_test.columns]

reg = LineReg(n_iter=100, learning_rate=lambda iter: 0.5 * (0.85 ** iter), metric='mse', sgd_sample=0.1)
reg.fit(X_train, y_train, verbose=10)
print(reg.predict(X_test).sum())