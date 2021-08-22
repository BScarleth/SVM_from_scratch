import numpy as np
from sklearn.metrics import classification_report


class SVM:
    def __init__(self, kernel="linear", d=1, max_iterations = 15, c = 1.0):
        self.kernel = kernel
        self.b = 0
        self.d = d
        self.c = c
        self.iters = max_iterations

    def fit(self, X, Y):
        self.X = np.copy(X)
        self.Y = np.copy(Y)
        self.m = len(self.X)

        current_alphas = np.zeros(self.m)
        E= np.zeros(self.m)
        tol = 1e-3
        passes = 0
        b = 0

        while passes < self.iters:
            num_changed_alphas = 0

            for i in range(self.m):
                E[i] = self.get_f_of_x(i, b, current_alphas) - Y[i]
                if (E[i] * Y[i] < -tol and current_alphas[i] < self.c) or (E[i] * Y[i] > tol and current_alphas[i] > 0):
                    samples = np.delete(np.arange(self.m), i)
                    j = np.random.choice(samples, size=1)[0]
                    E[j] = self.get_f_of_x(i, b, current_alphas) - self.Y[j]

                    old_alphas = np.copy(current_alphas)

                    if self.Y[i] != self.Y[j]:
                        L = np.maximum(0, current_alphas[j] - current_alphas[i])
                        H = np.minimum(self.c, self.c + current_alphas[j] - current_alphas[i])
                    else:
                        L = np.maximum(0, current_alphas[i] + current_alphas[j] - self.c)
                        H = np.minimum(self.c, current_alphas[i] + current_alphas[j])
                    if L == H:
                        continue

                    eta = 2 * self.apply_kernel(self.X[i], self.X[j]) - self.apply_kernel(self.X[i], self.X[i]) \
                               - self.apply_kernel(self.X[j], self.X[j])
                    if eta >= 0:
                        continue

                    current_alphas[j] -= self.Y[j] * (E[i] - E[j]) / eta
                    current_alphas[j] = self.set_alpha(current_alphas[j], L, H)
                    if np.abs(current_alphas[j] - old_alphas[j]) < tol:
                        continue

                    current_alphas[i] += self.Y[i] * self.Y[j] * (old_alphas[j] - current_alphas[j])

                    different_of_alphas_j = current_alphas[j] - old_alphas[j]
                    different_of_alphas_i = current_alphas[i] - old_alphas[i]

                    b1 = b - E[i] - (self.Y[i] * different_of_alphas_i * \
                         self.apply_kernel(self.X[i], self.X[i])) - (self.Y[j] \
                         * different_of_alphas_j * self.apply_kernel(self.X[i], self.X[j]))

                    b2 = b - E[j] - (self.Y[i] * different_of_alphas_i * \
                         self.apply_kernel(self.X[i], self.X[j])) - (self.Y[j] \
                         * different_of_alphas_j * self.apply_kernel(self.X[j], self.X[j]))

                    b = self.set_b(b1, b2, current_alphas[i], current_alphas[j])
                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        self.a = current_alphas
        self.b = b
        return self.a, self.b

    def set_b(self, b1, b2, alpha_i, alpha_j):
        if 0 < alpha_i < self.c:
            return b1
        if 0 < alpha_j < self.c:
            return b2
        return (b1 + b2)/2

    def set_alpha(self, alpha, L, H):
        if alpha > H:
            return H
        if alpha < L:
            return L
        return alpha

    def get_f_of_x(self, x_index, b, alpha):
        fx = 0
        for i in range(self.m):
            fx += alpha[i] * self.Y[i] * self.apply_kernel(self.X[i], self.X[x_index])
        fx += b
        return fx

    def apply_kernel(self, x1, x2):
        if self.kernel == "linear":
            return np.sum(np.multiply(x1, x2))
        if self.kernel == "poly":
            return np.power(np.sum(np.multiply(x1, x2)) + 1, self.d)

    def predict(self, attrs):
        fx = 0
        for i in range(self.m):
            fx += self.a[i] * self.Y[i] * self.apply_kernel(self.X[i], attrs)
        fx += self.b

        if fx >= 0:
           return 1.0
        else:
           return -1.0

    def print_parameters(self):
        print("kernel:", self.kernel, ", degree:", self.d,", C:", self.c, ", iter:", self.iters)

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

