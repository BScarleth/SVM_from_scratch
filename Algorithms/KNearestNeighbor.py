import numpy as np
import math

from sklearn.metrics import classification_report


class KNearest_Neighbor:
    def __init__(self, k, X, Y):
        self.k = k
        self.X = X
        self.Y = Y

    def euclidean_distance(self, p1, p2):
        return math.sqrt(np.sum((p1 - p2) ** 2))

    def get_neighbors(self, sample):
        distances = {}
        neighbors = {}
        for idx in range(len(self.X)):
            d = self.euclidean_distance(sample, self.X[idx])
            distances[d] = self.Y[idx]

        counter = 0
        for key, value in sorted(distances.items()):
            if counter >= self.k:
                break
            if value in neighbors:
                neighbors[value] += 1
            else:
                neighbors[value] = 1
            counter+=1
        return neighbors

    def predict(self, sample):
        neighbors = self.get_neighbors(sample)
        label = None
        max_number = -1
        for key, value in neighbors.items():
            if value > max_number:
                max_number = value
                label = key
        return label

    def score(self, X, Y):
        predictions = []
        for idx in range(len(X)):
            predictions.append(self.predict(X[idx]))
        print(classification_report(Y, predictions))
