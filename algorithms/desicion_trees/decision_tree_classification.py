import pandas as pd
import numpy as np

class MyTreeClf:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leaves = max_leafs
        self.tree = None # Структура дерева
        self.leafs_cnt = 0
        self.potential_leaves = 1
        self.current_depth = 0

    def __str__(self):
        return f'TreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leaves}'    

    def calculate_entropy(self, y: np.array):
        classes = np.unique(y)
        entropy = 0.0
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            if p > 0: # Игнорируем log(0)
                entropy -= p * np.log2(p)
        return entropy

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        col_name = None
        best_ig = float('-inf')
        split_value = None

        # Рассчитываем энтропию до разделения
        s0 = self.calculate_entropy(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(sorted(X.iloc[:, feature]))
            thresholds = [(thresholds[i] + thresholds[i+1])/2 for i in range(len(thresholds)-1)]
            for thresh in thresholds:
                left = X.iloc[:, feature] <= thresh
                right = X.iloc[:, feature] > thresh

                entropy_left  = self.calculate_entropy(y[left])
                entropy_right = self.calculate_entropy(y[right])

                n_left = sum(left)
                n_right = sum(right)
                n_total = n_left + n_right
                ig =  s0 - (n_left/n_total * entropy_left + n_right/n_total * entropy_right)

                if ig > best_ig:
                    best_ig = ig
                    col_name = X.columns[feature]
                    split_value = thresh
        return col_name, split_value, best_ig  
    
    def _create_leaf(self, y):
        return {'leaf': True, 'class_probabilities': np.sum(y==1) / len(y)}
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples = X.shape[0]
        if n_samples <= self.min_samples_split:
            self.leafs_cnt += 1
            self.potential_leaves -= 1
            return self._create_leaf(y)
        
        if depth >= self.max_depth:
            self.leafs_cnt += 1
            self.potential_leaves -= 1
            return self._create_leaf(y)
        
        if self.leafs_cnt + self.potential_leaves >= self.max_leaves and depth >= 1:
            self.leafs_cnt += 1
            self.potential_leaves -= 1
            return self._create_leaf(y)
        
        if len(y) == 1 or len(np.unique(y)) == 1:
            self.leafs_cnt += 1
            self.potential_leaves -= 1
            return self._create_leaf(y)

        # Ищем лучшее разделение
        best_feature, best_threshold, _ = self.get_best_split(X, y)

        # Если разделение невозможно, создаем лист
        if best_feature is None:
            self.leafs_cnt += 1
            self.potential_leaves -= 1
            return self._create_leaf(y)

        # Разделяем данные
        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold

        self.potential_leaves += 1

        # Рекурсивно строим левое и правое поддеревья
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def print_tree(self):
        print(self.tree)

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=50, n_features=20, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

tree = MyTreeClf(max_depth=3, min_samples_split=2, max_leafs=6)
#result = tree.get_best_split(X, y)
#print(tree.get_best_split(X, y))
tree.fit(X, y)
print(tree.leafs_cnt)
tree.print_tree()
