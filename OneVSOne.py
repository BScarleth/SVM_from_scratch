import numpy as np
import random

class OneVsOne:
    def __init__(self, list_of_clases, classifier, data):
        self.combinations = self.set_combinations(list_of_clases)
        self.classifier = classifier
        self.data = data

    def set_combinations(self, list_of_clases):
        combinations = []
        for i in range(0, len(list_of_clases) - 1):
            for j in range(i + 1, len(list_of_clases)):
                if list_of_clases[i] != list_of_clases[j]:
                    combinations.append([list_of_clases[i], list_of_clases[j]])
        return combinations

    def multiclass_classification(self):
        for pair in self.combinations:
            #svm.fit(X_train[:fraction], Y_train[:fraction], "linear")
            _data = self.data[pair[0]] + self.data[pair[1]]
            random.shuffle(_data)
            fraction = int(len(_data) * .80)

            X = _data[:][0]
            Y = _data[:][1]

            alpha, bias = self.classifier.fit(X[:fraction], Y[:fraction])

