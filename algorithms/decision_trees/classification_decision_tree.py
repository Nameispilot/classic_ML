import pandas as pd
import numpy as np

class Node:
    def __init__(self):
        self.feature = None
        self.value_split = None
        self.value_leaf = None
        self.side = None
        self.left = None
        self.right = None

class MyTreeClf:
    def __init__(self, max_depth: int=5, min_samples_split: int=2, max_leafs: int=20, bins=None, criterion: str='entropy', **kwargs):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion

        self.tree = None
        self.feature_splits = {}
        self.leafs_cnt = 1
        self.fi = {}
        self.sum_tree_values = 0
        self.N = kwargs['N'] if 'N' in kwargs else 0

    def __str__(self):
        return f'TreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}, criterion={self.criterion}'    

    def _entropy(self, y):
        classes = np.unique(y)
        entropy = 0.0
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            if p > 0: # Игнорируем log(0)
                entropy -= p * np.log2(p)
        return entropy
    
    def _gini_index(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        class_counts = np.bincount(y)
        probabilities = class_counts / n
        return 1.0 - np.sum(probabilities ** 2)

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        col_name = None
        best_ig = float('-inf')
        best_gini = float('+inf')
        best_gain = 0.0
        split_value = None

        # Рассчитываем энтропию до разделения
        if self.criterion == 'entropy':
            s0 = self._entropy(y)

        for feature in range(X.shape[1]):
            if self.bins is not None:
                thresholds = self.feature_splits[feature]
                if len(thresholds) == 0:
                    continue  # Пропускаем фичу, если нет разделителей
            else:
                thresholds = np.unique(sorted(X.iloc[:, feature]))
                thresholds = [(thresholds[i] + thresholds[i+1])/2 for i in range(len(thresholds)-1)]

            for thresh in thresholds:
                left = X.iloc[:, feature] <= thresh
                right = X.iloc[:, feature] > thresh

                if sum(left) == 0 or sum(right) == 0:
                    continue

                if self.criterion == 'entropy':
                    entropy_left  = self._entropy(y[left])
                    entropy_right = self._entropy(y[right])

                    n_left = len(y[left])
                    n_right = len(y[right])
                    n_total = n_left + n_right
                    ig =  s0 - (n_left/n_total * entropy_left + n_right/n_total * entropy_right)
                    if ig > best_ig:
                        best_ig = ig
                        best_gain = best_ig
                        col_name = X.columns[feature]
                        split_value = thresh

                elif self.criterion == 'gini':
                    gini_left = self._gini_index(y[left])
                    gini_right = self._gini_index(y[right])

                    n_left = len(y[left])
                    n_right = len(y[right])
                    n_total = n_left + n_right

                    gini = n_left/n_total*gini_left + n_right/n_total*gini_right
                    if gini < best_gini:
                        best_gini = gini
                        best_gain = best_gini
                        col_name = X.columns[feature]
                        split_value = thresh
                else:  
                    raise ValueError("Неправильно выбрано правило для узла!")      
        return col_name, split_value, best_gain

    def _calculate_feature_splits(self, X: pd.DataFrame):
        for feature in range(X.shape[1]):
            unique_values = np.unique(X.iloc[:, feature])
            if len(unique_values) <= self.bins - 1:
                self.feature_splits[feature] = unique_values
            else:
                _, bin_edges = np.histogram(X.iloc[:, feature], bins=self.bins)
                self.feature_splits[feature] = bin_edges[1:-1] # Исключаем крайние значения

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.tree = None
        if self.N == None:
            self.N = X.shape[0]
        self.fi = { col: 0 for col in X.columns }
        if self.bins is not None:
             self._calculate_feature_splits(X)
        self.tree = self.build_tree(self.tree, X, y)

    def build_tree(self, root, X_node, y_node, side='root', depth=0):
            if root is None:
                root = Node()
            col_name, best_threshold, ig = self.get_best_split(X_node, y_node)

            class_prob = 0
            if len(y_node):
                class_prob = len(y_node[y_node == 1]) / len(y_node) 

            # Если разделение невозможно (нет разделителей), создаем лист
            if best_threshold is None:
                root.side = side
                root.value_leaf = class_prob
                self.sum_tree_values += root.value_leaf 
                return root

            if len(y_node.unique()) == 1 or depth >= self.max_depth or \
                len(y_node) < self.min_samples_split or \
                (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs):
                root.side = side
                root.value_leaf = class_prob
                self.sum_tree_values += root.value_leaf 
                return root
            
            root.feature = col_name
            root.value_split = best_threshold
            self.leafs_cnt += 1
            
            self.fi[col_name] += ig * len(y_node) / self.N 

            left_indices = X_node[col_name] <= best_threshold
            right_indices = X_node[col_name] > best_threshold

            X_left, y_left = X_node.loc[left_indices], y_node[left_indices]
            X_right, y_right = X_node.loc[right_indices], y_node[right_indices]

            root.left = self.build_tree(root.left, X_left, y_left, 'left', depth + 1)
            root.right = self.build_tree(root.right, X_right, y_right, 'right', depth + 1)

            return root    
    
    def predict(self, X: pd.DataFrame):
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame):
        probabilities = [self._predict_single_proba(row, self.tree) for _, row in X.iterrows()] 
        return np.array(probabilities)

    def _predict_single_proba(self, x, tree):
        if tree.value_leaf is not None:
            return tree.value_leaf  # Вероятность первого класса
        if x[tree.feature] <= tree.value_split:
            return self._predict_single_proba(x, tree.left)
        else:
            return self._predict_single_proba(x, tree.right)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node.feature is not None:
            print(f"{'1.' * depth}{node.feature} > {node.value_split}")
            if node.left is not None:
                self.print_tree(node.left, depth + 1)
            if node.right is not None:
                self.print_tree(node.right, depth + 1)
        else:
            print(f"{'1.' * depth}{node.side} = {node.value_leaf}")
