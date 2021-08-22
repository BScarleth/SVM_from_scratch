from sklearn.metrics import classification_report

from Algorithms.NN import NN
from Helper import Helper
import random

import numpy as np

#Reading data features
data_directory = "../data/"
arabic = Helper.read_features_file(data_directory + "arabic")[:100]
english = Helper.read_features_file(data_directory + "english")[:100]
spanish = Helper.read_features_file(data_directory + "spanish")[:100]

data = arabic + english + spanish

random.shuffle(data)
X = []
Y = []
for _d in data:
    extra_dim = np.expand_dims(_d["features"], axis=2)
    X.append(extra_dim)
    Y.append(_d["audio_class"])

X = np.array(X)
Y = np.array(Y)
fraction = int(len(X) * 80 / 100)

nn = NN(3)
nn.fit(X[:fraction], Y[:fraction], epochs=550)
score, predictions = nn.score(X[fraction:], Y[fraction:])

print(classification_report(Y[fraction:], predictions))