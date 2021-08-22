from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import callbacks
import numpy as np
import matplotlib.pyplot as plt

class NN:
    def __init__(self, number_classes):
        #self.opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
        self.model = self.get_model(number_classes)


    def get_model(self, number_classes):
        #initializer = tf.keras.initializers.VarianceScaling(scale=2)

        model = keras.Sequential(
            [
                layers.InputLayer(input_shape=(13, 862, 1)),
                layers.Flatten(),
                layers.Dense(250, activation="relu"), #100
                layers.Dropout(.2),
                layers.Dense(250, activation="relu"),
                layers.Dense(number_classes, activation="softmax"),
            ]
        )

        """
        model = keras.Sequential(
            [
                layers.InputLayer(input_shape=(13, 862, 1)),
                layers.Conv2D(filters=8, kernel_size=(1, 50), activation="relu"),
                layers.MaxPooling2D((2, 2), strides=2),
                #layers.Conv2D(filters=64, kernel_size=(1, 50), activation="relu"),
                #layers.MaxPooling2D((2, 2), strides=2),
                layers.Flatten(),
                layers.Dense(100, activation="relu"),
                layers.Dense(number_classes, activation="softmax"),
            ]
        )
        """

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=self.opt,
                      metrics=['accuracy'])

        return model

    def fit(self, X, Y, epochs):
        records = self.model.fit(X, Y, epochs= epochs, validation_split=0.2)

        plt.plot(records.history['accuracy'], '-o')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train'])
        plt.title('Train accuracy')
        plt.show()

        plt.plot(records.history['val_accuracy'], '-o')
        plt.xlabel('epoch')
        plt.ylabel('val_accuracy')
        plt.legend(['Train'])
        plt.title('Train val_accuracy')
        plt.show()

        print("history")
        plt.plot(records.history['loss'], '-o')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train'])
        plt.title('Train Losses')
        plt.yscale('log')
        plt.show()


    def predict(self, sample):
        scores = self.model.predict(np.expand_dims(sample,0))
        prediction = np.argmax(scores)
        return prediction

    def score(self, X, Y):
        count = 0
        predictions  = []
        for idx in range(len(X)):
            p = self.predict(X[idx])
            predictions.append(p)
            #print("p: ", p, " ---> real: ", Y[idx])
            if p == Y[idx]:
                count += 1
        accuracy = (count * 100) / len(X)
        print("Acurracy: ", accuracy)
        return (count * 100) / len(X), predictions
