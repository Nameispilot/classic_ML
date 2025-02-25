import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MyKMeans():
    def __init__(self, n_clusters: int = 3, max_iter: int = 10, n_init : int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = []
        self.inertia_ = np.inf

    def __str__(self):
        return f"MyKMeans class: n_clusters={self.n_clusters}, max_iter={self.max_iter}, n_init={self.n_init}, random_state={self.random_state}"    
    
    def euclidean_dist(self, centroids, row: pd.Series):
        distance = np.sum((row.values - centroids) ** 2, axis=1) ** 0.5  
        return distance
    
    def calc_centroids(self, centroids: list, X: pd.DataFrame, clasters: list):
        for i, idxs in enumerate(clasters):
            X_ = X.iloc[idxs]
            if X_.empty:
                continue
            centroids[i] = np.mean(X_, axis=0).tolist()       
        return centroids   

    def fit(self, X: pd.DataFrame):
        clasters = []
        np.random.seed(seed=self.random_state)
        max_ = X.max(axis=0).tolist()
        min_ = X.min(axis=0).tolist()
        for _ in range(self.n_init):
            ''' Случайно полученные координаты центроидов'''
            centroids = []
            for _ in range(self.n_clusters):
                centroids.append([np.random.uniform(min_[j],max_[j]) for j in range(X.shape[1])]) 

            for i in range(self.max_iter):
                distances = X.apply(lambda row: self.euclidean_dist(centroids, row), axis=1, result_type='expand')
                
                ''' Хранит в себе индексы точек, принадлежающие кластерам по возрастанию'''
                clasters = distances.idxmin(axis="columns")
                clasters = [clasters.index[clasters == i].tolist() for i in range(self.n_clusters)]

                '''Перерасчет центриодов кластеров'''
                centroids = self.calc_centroids(centroids, X, clasters)
                
            ''' Выбор лучшего набора'''    
            wcss = 0.0
            for i, idxs in enumerate(clasters):
                X_ = X.iloc[idxs]
                wcss += np.sum(np.sum((X_ - centroids[i]) ** 2)) 
            if wcss <= self.inertia_:
                self.inertia_ = wcss
                self.cluster_centers_ = centroids     
                
    def predict(self, X: pd.DataFrame):
        distances = X.apply(lambda row: self.euclidean_dist(self.cluster_centers_, row), axis=1, result_type='expand')
        clasters = distances.idxmin(axis="columns")
        return clasters


from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=5, n_features=5, cluster_std=2.5, random_state=42)
X = pd.DataFrame(X)
X.columns = [f'col_{col}' for col in X.columns]

knn = MyKMeans(10, 10, 3)
knn.fit(X)
knn.predict(X)