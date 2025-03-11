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

class TreeReg :
    def __init__(self, max_depth:int=5, min_samples_split:int=2, max_leafs:int=20, bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = None
        self.bins = bins

        self.tree =  None
        self.leafs_cnt = 1
        self.feature_splits = {}
        self.fi = {}
        self.sum_tree_values = 0

    def __str__(self):
        return f'TreeReg  class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'    

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):   
        col_name = None
        split_value = None
        best_mse = float('-inf')

        mse_p = self._mse(y)
        for feature in range(X.shape[1]):
            if self.bins is not None:
                thresholds = self.feature_splits[feature]
                if len(thresholds) == 0:
                    continue  # Пропускаем фичу, если нет разделителей
            else:
                thresholds = np.unique(sorted(X.iloc[:, feature]))
                thresholds = [(thresholds[i] + thresholds[i+1])/2 for i in range(len(thresholds)-1)]

            n = len(y)
            for thresh in thresholds:
                left = X.iloc[:, feature] <= thresh
                right = X.iloc[:, feature] > thresh

                if sum(left) == 0 or sum(right) == 0:
                    continue

                mse_left = self._mse(y[left])
                mse_right = self._mse(y[right])

                n_left = len(y[left])
                n_right = len(y[right])

                mse_t = mse_p - (n_left/n*mse_left + n_right/n*mse_right)
                if mse_t > best_mse:
                    col_name = X.columns[feature]
                    #col_name = feature
                    split_value = thresh
                    best_mse = mse_t
        return col_name, split_value   

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
        self.N = X.shape[0]
        # Инициализация важности фичей
        for feature in X.columns:
            self.fi[feature] = 0.0
        if self.bins is not None:
             self._calculate_feature_splits(X)
        self.tree = self.build_tree(self.tree, X, y)

    def build_tree(self, root, X_node, y_node, side='root', depth=0):
        if root is None:
            root = Node()
        col_name, best_threshold = self.get_best_split(X_node, y_node)   

        # Если разделение невозможно (нет разделителей), создаем лист
        if best_threshold is None:
            root.side = side
            root.value_leaf = np.mean(y_node)
            self.sum_tree_values += root.value_leaf 
            return root

        if len(y_node.unique()) == 1 or depth >= self.max_depth or \
            len(y_node) < self.min_samples_split or \
            (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs):
            root.side = side
            root.value_leaf = np.mean(y_node)
            self.sum_tree_values += root.value_leaf 
            return root
            
        root.feature = col_name
        root.value_split = best_threshold
        self.leafs_cnt += 1
            
        left_indices = X_node[col_name] <= best_threshold
        right_indices = X_node[col_name] > best_threshold

        X_left, y_left = X_node.loc[left_indices], y_node[left_indices]
        X_right, y_right = X_node.loc[right_indices], y_node[right_indices]

        root.left = self.build_tree(root.left, X_left, y_left, 'left', depth + 1)
        root.right = self.build_tree(root.right, X_right, y_right, 'right', depth + 1)

        I = self._mse(y_node)
        I_l = self._mse(y_node[left_indices])
        I_r = self._mse(y_node[right_indices])    

        N_p = X_node.shape[0]
        N_l = len(X_node[left_indices])
        N_r = len(X_node[right_indices])

        FI = (N_p / self.N) * (I - (N_l / N_p) * I_l - (N_r / N_p) * I_r)
        self.fi[col_name] += FI
        
        return root   
     
    def predict(self, X: pd.DataFrame):
        logits = [self._predict_single_proba(row, self.tree) for _, row in X.iterrows()] 
        return np.array(logits)

    def _predict_single_proba(self, x, tree):
        if tree.value_leaf is not None:
            return tree.value_leaf
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

'''from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
X, y = data['data'], data['target']

tree = TreeReg(max_depth=3, min_samples_split=2, max_leafs=5, bins=None)

tree.fit(X, y)
print(tree.leafs_cnt)
print(tree.sum_tree_values)

tree.predict(X)'''
