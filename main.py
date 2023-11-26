import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

class FuzzyCluster:
    def __init__(self, data):
        self.data = data
        self.membership = np.ones(len(self.data))

    def update_membership(self, centroids, m):
        distances = cdist(self.data, centroids)
        self.membership = 1 / np.sum((distances / distances[:, None]) ** (2 / (m - 1)), axis=1)

    def update_centroid(self):
        self.centroid = np.average(self.data, axis=0, weights=self.membership)

    def __str__(self):
        return f"\nMembership: {self.membership}\nData: {self.data}\nCentroid: {self.centroid}"

class FuzzyClustering:
    def __init__(self, data, k=4, m=2):
        self._k = k
        self._m = m
        self.data = data
        self.clusters = [FuzzyCluster(data) for _ in range(k)]

    def update_clusters(self):
        centroids = [cluster.centroid for cluster in self.clusters]
        for cluster in self.clusters:
            cluster.update_membership(centroids, self._m)
            cluster.update_centroid()

    def iterate(self, iterations=20):
        for _ in range(iterations):
            self.update_clusters()

    def __str__(self):
        return '\n'.join([f"Cluster {i}: {cluster}" for i, cluster in enumerate(self.clusters)])

class FuzzyClusteringApp:
    def __init__(self):
        self.points = np.concatenate([
            np.random.randint(0, high=10, size=(2, 1)) for _ in range(25)
        ] + [
            np.random.randint(15, high=25, size=(2, 1)) for _ in range(25)
        ], axis=1)

        self.clustering = FuzzyClustering(self.points, k=2)

        self.root = tk.Tk()
        self.root.title("Fuzzy Clustering App")

        self.iterate_button = tk.Button(self.root, text="Iterate", command=self.iterate)
        self.iterate_button.pack()

        self.reload_button = tk.Button(self.root, text="Reload", command=self.reload)
        self.reload_button.pack()

        self.add_data_button = tk.Button(self.root, text="Add Data", command=self.add_data)
        self.add_data_button.pack()

        self.plot_button = tk.Button(self.root, text="Plot", command=self.plot)
        self.plot_button.pack()

    def iterate(self):
        self.clustering.iterate()
        print(self.clustering)

    def reload(self):
        self.clustering = FuzzyClustering(self.points, k=2)
        print("Reloaded Clustering:")
        print(self.clustering)

    def add_data(self):
        new_data = np.concatenate([np.random.randint(5, high=20, size=(2, 1)) for _ in range(10)], axis=1)
        self.clustering.data = np.concatenate((self.clustering.data, new_data), axis=1)
        self.clustering.clusters = [FuzzyCluster(self.clustering.data) for _ in range(self.clustering._k)]
        print(f"Added Data: {new_data}")
        print(self.clustering)

    def plot(self):
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 5))
        for i, cluster in enumerate(self.clustering.clusters):
            plt.scatter(cluster.data[:, 0], cluster.data[:, 1], label=f'Cluster {i}')

        centroids = np.array([cluster.centroid for cluster in self.clustering.clusters])
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', label='Centroids')

        plt.title('Fuzzy Clustering')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.show()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FuzzyClusteringApp()
    app.run()
#REFERENCE FROM FUZZZYMEANS COPIED SOME PART 
