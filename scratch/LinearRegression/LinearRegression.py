import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.001, n_inters = 1000):
        self.lr = lr
        self.n_inters = n_inters
        self.weigths = None
        self.bias = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weigths = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_inters):
            y_pred = np.dot(X, self.weigths) + self.bias # y_hat = wX + b -> [w1x1+b, w2x2 + b ... wnxn + b]


            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) 
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weigths = self.weigths - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self,X):
        y_pred = np.dot(X, self.weigths) + self.bias
        return y_pred   