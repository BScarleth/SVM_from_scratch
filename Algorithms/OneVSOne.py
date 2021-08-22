import numpy as np
import random
from sklearn.metrics import classification_report

class OneVsOne:
    def __init__(self, list_of_clases, classifier, data_training, data_testing):
        self.list_of_clases = list_of_clases
        self.combinations = self.set_combinations(list_of_clases)
        self.classifier = classifier
        self.data_training = data_training
        self.data_testing = data_testing
        self.test_elements = len(data_testing)

    def set_combinations(self, list_of_clases):
        combinations = []
        for i in range(0, len(list_of_clases) - 1):
            for j in range(i + 1, len(list_of_clases)):
                if list_of_clases[i] != list_of_clases[j]:
                    combinations.append([list_of_clases[i], list_of_clases[j]])
        return combinations

    def get_final_class(self, predicted_values):
        final_classes = []
        for p in predicted_values:
            final_classes.append(np.argmax(p))
        return final_classes

    def score(self, predictions, Y):
        print(classification_report(Y, predictions))

    def multiclass_classification(self):
        predicted_values = np.zeros((self.test_elements, len(self.list_of_clases)))

        X_t = []
        Y_t = []
        for _d in self.data_testing:
            X_t.append(_d[0])
            Y_t.append(_d[1])

        for pair in self.combinations:
            print("Processing classifier for: ", pair)
            _data_training = self.data_training[pair[0]] + self.data_training[pair[1]]
            random.shuffle(_data_training)

            X = []
            Y = []
            for _d in _data_training:
                X.append(_d[0])
                Y.append(_d[1])

            self.classifier.fit(X, Y)
            self.classifier.evaluate(X_t)
            for prediction in range(len(self.classifier.pred)):
                for label in range(len(self.list_of_clases)):
                    if pair[0] == label and self.classifier.pred[prediction] == 1:
                        predicted_values[prediction, label] +=1
                    elif pair[1] == label and self.classifier.pred[prediction] == -1:
                        predicted_values[prediction, label] += 1

        final_predictions = self.get_final_class(predicted_values)
        self.score(final_predictions, Y_t)


