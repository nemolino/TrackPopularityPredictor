import numpy as np
from math import e
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel

class KRR(BaseEstimator, RegressorMixin):

    def __init__(self, λ = 1.0, γ = 1.0):

        if λ < 0: raise ValueError("λ must be >= 0")
        if γ <= 0: raise ValueError("γ must be > 0")
        self.λ = λ
        self.γ = γ

        self.X_ = None
        self.c_ = None 
        
    def fit(self, X, y):

        self.X_ = X.values
        K = rbf_kernel(self.X_, self.X_, gamma=1/(self.γ * 2))
        self.c_ = y.T @ np.linalg.inv(K + self.λ * np.identity(self.X_.shape[0]))

    def predict(self, X):

        if self.c_ is None or self.X_ is None:
            raise RuntimeError('Model is still to fit')
        
        return self.c_ @ rbf_kernel(self.X_, X.values, gamma=1/(self.γ * 2))
    
    """ totally from scratch but slow version

    def gaussian_kernel(self, x_i, x_j):
        euclidean_distance = np.linalg.norm(x_i - x_j)
        #return e ** ((euclidean_distance * euclidean_distance) / (-2 * self.γ))
        return self.base_ ** (euclidean_distance * euclidean_distance)

    def fit(self, X, y):

        self.X_ = X.values
        m = self.X_.shape[0]

        K = np.zeros((m, m))
        for i in range(m):
            for j in range(i,m): 
                K[i][j] = self.gaussian_kernel(self.X_[i, :], self.X_[j, :])
                K[j][i] = K[i][j]

        self.c_ = y.T @ np.linalg.inv(K + self.λ * np.identity(m))

    def predict(self, X):

        if self.c_ is None or self.X_ is None:
            raise RuntimeError('Model is still to fit')
        
        X_ = X.values
        
        n = X_.shape[0]
        y_prediction = np.zeros(n)

        m = self.X_.shape[0]
        k = np.zeros(m)

        base = e ** (1 / (-2 * self.γ))

        for j in range(n):
            for i in range(m):
                euclidean_distance = np.linalg.norm(self.X_[i, :] - X_[j, :])
                k[i] = base ** (euclidean_distance * euclidean_distance)
            y_prediction[j] = self.c_ @ k

        return y_prediction

    """