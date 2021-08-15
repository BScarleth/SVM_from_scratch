import numpy as np
from random import choice
import random


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

        max_delta_E = 0  # self.tol
        if (r2 < -self.tol and alpha2 < self.c) or (r2 > self.tol and alpha2 > 0):

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

            random.shuffle(indexes)
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
            L = np.maximum(0, alpha2 - alpha1)
            H = np.minimum(self.c, self.c + alpha2 - alpha1)
        else:
            L = np.maximum(0, alpha2 + alpha1 - self.c)
            H = np.minimum(self.c, alpha2 + alpha1)
        if L == H:
            return 0

        eta = self.apply_kernel(self.X[i1], self.X[i1]) + self.apply_kernel(self.X[i2], self.X[i2]) \
                   - 2 * self.apply_kernel(self.X[i1], self.X[i2])

        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            a2 = self.set_alpha(a2, L, H)
        else:
            # checkkkk
            c1 = eta / 2.0
            c2 = y2 * (E1 - E2) - eta * alpha2
            Lobj = c1 * L * L + c2 * L
            Hobj = c1 * H * H + c2 * H
            if Lobj < Hobj - self.tol:
                a2 = L
            elif Lobj > Hobj + self.tol:
                a2 = H
            else:
                a2 = alpha2
            # checkkkk

        if np.abs(a2 - alpha2) < self.tol * (a2 + alpha2 + self.tol):
            return 0

        a1 = alpha1 + s * (alpha2 - a2)

        different_of_alphas_a2 = a2 - alpha2
        different_of_alphas_a1 = a1 - alpha1

        b1 = self.b + E1 + (y1 * different_of_alphas_a1 * self.apply_kernel(self.X[i1], self.X[i1])) \
             + (y2 * different_of_alphas_a2 * self.apply_kernel(self.X[i1], self.X[i2]))

        b2 = self.b + E2 + (y1 * different_of_alphas_a1 * self.apply_kernel(self.X[i1], self.X[i2])) \
             + (y2 * different_of_alphas_a2 * self.apply_kernel(self.X[i2], self.X[i2]))

        _b = self.set_b(b1, b2, a1, a2)
        #############
        t1 = y1 * (a1 - alpha1)
        t2 = y2 * (a2 - alpha2)
        dw = _b - self.b

        for i in range(self.m):
            if (self.current_alphas[i] > 0) and (self.current_alphas[i] < self.c):
                self.E[i] += t1 * self.apply_kernel(self.X[i1], self.X[i]) \
                             + t2 * self.apply_kernel(self.X[i2], self.X[i]) - dw

        self.E[i1] = 0
        self.E[i2] = 0

        ################
        self.b = _b
        self.current_alphas[i1] = a1
        self.current_alphas[i2] = a2
        return 1

    def calculate_error_e(self, idx):
        if 0 < self.current_alphas[idx] < self.c:
            return self.E[idx]
        return self.get_f_of_x(idx)

    def set_b(self, b1, b2, alpha1, alpha2):
        if 0 < alpha1 < self.c:
            return b1
        if 0 < alpha2 < self.c:
            return b2
        return (b1 + b2) / 2

    def set_alpha(self, alpha, L, H):
        if alpha > H:
            return H
        if alpha < L:
            return L
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
        count = 0
        for idx in range(len(X)):
            p = self.predict(X[idx])
            if p == Y[idx]:
                count += 1
        accuracy = (count * 100) / len(X)
        self.print_parameters()
        print("Acurracy: ", accuracy)
        return (count * 100) / len(X)
