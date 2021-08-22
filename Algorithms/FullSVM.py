import numpy as np
import random
from sklearn.metrics import classification_report


class SVM:
    def __init__(self, kernel="linear", d=1, c=1.0):
        self.kernel = kernel
        self.b = 0
        self.d = d
        self.c = c
        self.tol = 1e-3

    def fit(self, X, Y):
        self.X = np.copy(X)
        self.Y = np.copy(Y)
        self.m = len(self.X)
        self.E = np.zeros(self.m)
        self.current_alphas = np.zeros(self.m)

        self.main_routine()
        return self.current_alphas, self.b

    def main_routine(self):
        print("SVM starting...")
        num_changed_alphas = 0
        examineAll = 1

        while num_changed_alphas > 0 or examineAll:
            num_changed_alphas = 0
            if examineAll == 1:
                for i2 in range(self.m):
                    num_changed_alphas += self.examineExample(i2)
            else:
                for i2 in range(self.m):
                    if self.current_alphas[i2] > 0 and self.current_alphas[i2] < self.c:
                        num_changed_alphas += self.examineExample(i2)
            if examineAll == 1:
                examineAll = 0
            elif num_changed_alphas == 0:
                examineAll = 1

    def examineExample(self, i2):
        y2 = self.Y[i2]
        alpha2 = self.current_alphas[i2]
        E2 = self.calculate_error_e(i2)

        r2 = E2 * y2

        if (r2 < -self.tol and alpha2 < self.c) or (r2 > self.tol and alpha2 > 0):
            max_delta_E = 0
            # checkkkk
            i1 = -1
            for i in range(self.m):
                if self.current_alphas[i] > 0 and self.current_alphas[i] < self.c:
                    del_E = np.abs(self.calculate_error_e(i) - E2)
                    if del_E > max_delta_E:
                        max_delta_E = del_E
                        i1 = i
            if i1 >= 0:
                if self.take_step(i1, i2):
                    return 1
            # checkkkk

            indexes = list(range(self.m))
            random.shuffle(indexes)

            for i1 in indexes:
                if self.current_alphas[i1] > 0 and self.current_alphas[i1] < self.c:
                    if self.take_step(i1, i2):
                        return 1

            for i1 in indexes:
                if self.take_step(i1, i2):
                    return 1
        return 0

    def take_step(self, i1, i2):
        if i1 == i2:
            return 0
        alpha1 = self.current_alphas[i1]
        alpha2 = self.current_alphas[i2]
        y1 = self.Y[i1]
        y2 = self.Y[i2]
        E1 = self.calculate_error_e(i1)
        E2 = self.calculate_error_e(i2)
        s = y1 * y2

        if y1 != y2:
            self.L = np.maximum(0, alpha2 - alpha1)
            self.H = np.minimum(self.c, self.c + alpha2 - alpha1)
        else:
            self.L = np.maximum(0, alpha2 + alpha1 - self.c)
            self.H = np.minimum(self.c, alpha2 + alpha1)
        if self.L == self.H:
            return 0

        eta = self.apply_kernel(self.X[i1], self.X[i1]) + self.apply_kernel(self.X[i2], self.X[i2]) \
                   - 2 * self.apply_kernel(self.X[i1], self.X[i2])

        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            a2 = self.set_alpha(a2)
        else:
            # checkkkk
            c1 = eta / 2.0
            c2 = y2 * (E1 - E2) - eta * alpha2
            Lobj = c1 * self.L * self.L + c2 * self.L
            Hobj = c1 * self.H * self.H + c2 * self.H
            if Lobj < Hobj - self.tol:
                a2 = self.L
            elif Lobj > Hobj + self.tol:
                a2 = self.H
            else:
                a2 = alpha2
            # checkkkk

        if np.abs(a2 - alpha2) < self.tol * (a2 + alpha2 + self.tol):
            return 0

        a1 = alpha1 + s * (alpha2 - a2)


        b1 = self.b + E1 + y1 * (a1 - alpha1) * self.apply_kernel(self.X[i1], self.X[i1]) + y2 * (a2 - alpha2) * self.apply_kernel(self.X[i1], self.X[i2])
        b2 = self.b + E2 + y1 * (a1 - alpha1) * self.apply_kernel(self.X[i1], self.X[i2]) + y2 * (a2 - alpha2) * self.apply_kernel(self.X[i2], self.X[i2])

        _b = self.set_b(b1, b2, a1, a2)
        t1 = y1 * (a1 - alpha1)
        t2 = y2 * (a2 - alpha2)
        dw = _b - self.b

        for i in range(self.m):
            if (self.current_alphas[i] > 0) and (self.current_alphas[i] < self.c):
                self.E[i] += t1 * self.apply_kernel(self.X[i1], self.X[i]) + t2 * self.apply_kernel(self.X[i2], self.X[i]) - dw

        self.E[i1] = 0
        self.E[i2] = 0

        self.b = _b
        self.current_alphas[i1] = a1
        self.current_alphas[i2] = a2
        return 1

    def calculate_error_e(self, idx):
        if self.current_alphas[idx]  > 0 and self.current_alphas[idx] < self.c:
            return self.E[idx]
        return self.get_f_of_x(idx)

    def set_b(self, b1, b2, alpha1, alpha2):
        if alpha1 > 0 and alpha1 < self.c:
            return b1
        if alpha2 > 0 and alpha2 < self.c:
            return b2
        return (b1 + b2) / 2

    def set_alpha(self, alpha):
        if alpha < self.L:
            return self.L
        if alpha > self.H:
            return self.H
        return alpha

    def get_f_of_x(self, x_index):
        fx = 0
        for i in range(self.m):
            fx += self.current_alphas[i] * self.Y[i] * self.apply_kernel(self.X[i], self.X[x_index])
        fx -= self.b
        return fx - self.Y[x_index]

    def apply_kernel(self, x1, x2):
        if self.kernel == "linear":
            return np.sum(np.multiply(x1, x2)) + 1  # check if 1 is needed
        if self.kernel == "poly":
            return np.power(np.sum(np.multiply(x1, x2)) + 1, self.d)

    def predict(self, attrs):
        fx = 0
        for i in range(self.m):
            fx += self.current_alphas[i] * self.Y[i] * self.apply_kernel(self.X[i], attrs)
        fx -= self.b

        if fx >= 0:
            return 1.0
        else:
            return -1.0

    def print_parameters(self):
        print("kernel:", self.kernel, ", degree:", self.d,", C:", self.c)

    def score(self, X, Y):
        predictions = []
        for idx in range(len(X)):
            predictions.append(self.predict(X[idx]))
        self.print_parameters()
        print(classification_report(Y, predictions))

    def evaluate(self, X):
        self.pred = []
        for idx in range(len(X)):
            self.pred.append(self.predict(X[idx]))
