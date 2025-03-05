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
    def __init__(self, max_depth: int = 5, min_samples_split: int =2, max_leafs: int =20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1

        self.sum_tree_values = 0

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
                    #col_name = feature
                    split_value = thresh
        return col_name, split_value, best_ig  
    
    def fit(self, X, y):
        self.tree = None
        self.tree = self.build_tree(self.tree, X, y)

    def build_tree(self, root, X_node, y_node, side='root', depth=0):
            if root is None:
                root = Node()
            col_name, best_threshold, _ = self.get_best_split(X_node, y_node)

            class_prob = 0
            if len(y_node):
                class_prob = len(y_node[y_node == 1]) / len(y_node)  

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

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

'''df = pd.read_csv('banknote+authentication.zip', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:,:4], df['target']'''

tree = MyTreeClf(max_depth=15, min_samples_split=20, max_leafs=30)

tree.fit(X, y)
print(tree.leafs_cnt)
print(tree.sum_tree_values)
tree.print_tree()

print(tree.predict(X))
print(tree.predict_proba(X))
