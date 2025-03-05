import pandas as pd
import numpy as np

class MyAgglomerative():
    def __init__(self, n_clusters: int = 3, metric: str ='euclidean'):
        self.n_clusters = n_clusters 
        self.metric = metric

    def __str__(self):
        return f'MyAgglomerative class: n_clusters={self.n_clusters}'
    
    def _euclidean_dist(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def _chebyshev_dist(self, point1, point2):
        return np.max(np.abs(point1 - point2))
    
    def _manhattan_dist(self, point1, point2):
        return np.sum(np.absolute(point1 - point2))
    
    def _cosine_dist(self, point1, point2):
        dot_product = np.dot(point1, point2)
        norm1 = np.linalg.norm(point1)
        norm2 = np.linalg.norm(point2)
        return 1 - dot_product / (norm1 * norm2)
    
    def fit_predict(self, X: pd.DataFrame):
        points = np.array(X.values)
        n = len(points) 

        # Инициализация: каждая точка — отдельный кластер
        # clusters: список, где каждый элемент — это список индексов точек, принадлежащих кластеру
        clusters = [[i] for i in range(n)]
        # centroids: список центроидов кластеров
        centroids = [points[i] for i in range(n)]

        # Вычисляем матрицу расстояний между всеми парами точек
        distances = np.zeros((n, n))
        for i in range(n):
                for j in range(i + 1, n):
                    distances[i, j] = getattr(self, f'_{self.metric}_dist')(points[i], points[j])
                    distances[j, i] = distances[i, j]  # Матрица симметрична
        np.fill_diagonal(distances, np.inf)  # Игнорируем диагональ             

        while len(clusters) > self.n_clusters:
            min_dist  = np.min(distances)
            # Находим индексы точек с минимальным расстоянием
            cluster1, cluster2 = np.where(distances == min_dist)[0]
            #print(min_dist, cluster1, cluster2)

            # Объединяем два ближайших кластера
            clusters[cluster1].extend(clusters[cluster2])
            del clusters[cluster2]

            # Пересчитываем центроид нового кластера
            new_centroid = np.mean([points[i] for i in clusters[cluster1]], axis=0)
            centroids[cluster1] = new_centroid
            del centroids[cluster2]
            
            # Обновляем матрицу расстояний
            distances = np.delete(distances, cluster2, axis=0)
            distances = np.delete(distances, cluster2, axis=1)

            for i in range(len(clusters)):
                if i != cluster1:
                    distances[i, cluster1] = getattr(self, f'_{self.metric}_dist')(centroids[i], centroids[cluster1])
                    distances[cluster1, i] = distances[i, cluster1]
            np.fill_diagonal(distances, np.inf)  
        # Создаем массив меток кластеров
        labels = np.zeros(n, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_id in cluster:
                labels[point_id] = cluster_id
    
        return labels          

from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=100, centers=10, n_features=5, cluster_std=2.5, random_state=42)
X = pd.DataFrame(X)
X.columns = [f'col_{col}' for col in X.columns]

agg = MyAgglomerative(n_clusters=10, metric='cosine')
labels = agg.fit_predict(X)
print(labels)