from Algorithms.KNearestNeighbor import KNearest_Neighbor
from Helper import Helper
import random
import numpy as np


#Reading data features
data_directory = "../data/"
arabic = Helper.read_features_file(data_directory + "arabic")[:100]
english = Helper.read_features_file(data_directory + "english")[:100]
spanish = Helper.read_features_file(data_directory + "spanish")[:100]

#Preparing data for testing data
data = english + spanish + arabic
random.shuffle(data)

X = []
Y = []
for _d in data:
    features = np.mean(_d["features"].T, axis=0)
    X.append(features)
    Y.append(_d["audio_class"])

fraction = int(len(X) * 80 / 100)

#Testing the K-nearest-neighbor algorithm
k_nn = KNearest_Neighbor(9, X[:fraction], Y[:fraction])
print(k_nn.score(X[fraction:], Y[fraction:]))
