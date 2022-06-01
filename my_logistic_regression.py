import numpy as np


class MyLogisticRegression():
    """Description:
My personnal logistic regression to classify things.
"""
    def __init__(self, theta, alpha=0.001, max_iter=1000,
                 penalty='l2', lambda_=0.5):
        error_msg = "theta has to be an numpy.array or list,\
 a vector."
        if isinstance(theta, np.ndarray):
            if theta.ndim != 2 or theta.shape[1] != 1 or not theta.size\
                    or not np.issubdtype(theta.dtype, np.number):
                print(error_msg)
                return None
        elif isinstance(theta, list):
            try:
                theta = np.array(theta).reshape((-1, 1))
                assert np.issubdtype(theta.dtype, np.number)
            except Exception:
                print(error_msg)
                return None
        else:
            print(error_msg)
            return None
        if not isinstance(alpha, (float, int)):
            print("alpha has to be a float.")
            return None
        if alpha <= 0:
            print("The learning rate has to be strictly positive.")
            return None
        if not isinstance(max_iter, int):
            print("max_iter has to be an int.")
            return None
        if max_iter < 0:
            print("The number of iterations has to be positive.")
            return None
        if penalty not in ['l2', 'none']:
            print('penalty has to be either `l2` or `none`.')
            return None
        if not isinstance(lambda_, (float, int)):
            print("lambda_ has to be a float.")
            return None
        if lambda_ < 0:
            print("lambda_ has to be positive.")
            return None
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = theta
        self.penalty = penalty
        self.lambda_ = lambda_

    def sigmoid_(self, x):
        if not isinstance(x, np.ndarray) or not x.size\
                or not np.issubdtype(x.dtype, np.number):
            print("x has to be an numpy.array, a vector.")
            return None
        return (1 / (1 + np.exp(-x))).reshape(-1, 1)

    def fit_(self, x, y):
        if not hasattr(self, 'thetas') or not hasattr(self, 'alpha')\
                or not hasattr(self, 'max_iter')\
                or not hasattr(self, 'penalty')\
                or not hasattr(self, 'lambda_'):
            return None
        if not isinstance(x, np.ndarray) or x.ndim != 2\
                or not x.size or not np.issubdtype(x.dtype, np.number):
            print("x has to be an numpy.array, a matrix of shape m * n.")
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
                or not y.size or not np.issubdtype(y.dtype, np.number):
            print("y has to be an numpy.array, a vector of shape m * 1.")
            return None
        if x.shape[0] != y.shape[0]:
            print('x and y must have the same number of rows.')
            return None
        if self.thetas.shape[0] != x.shape[1] + 1:
            print("x and theta's shapes don't match.")
            return None
        X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        m = x.shape[0]
        if self.penalty == 'none':
            for _ in range(self.max_iter):
                predictions = self.sigmoid_(X @ self.thetas)
                grad = (X.T @ (predictions - y)) / m
                self.thetas = self.thetas - self.alpha * grad
        elif self.penalty == 'l2':
            for _ in range(self.max_iter):
                predictions = self.sigmoid_(X @ self.thetas)
                theta_prime = self.thetas.copy()
                theta_prime[0, 0] = 0
                grad = (X.T @ (predictions - y) +
                        self.lambda_ * theta_prime) / m
                self.thetas = self.thetas - self.alpha * grad

    def predict_(self, x):
        if not hasattr(self, 'thetas'):
            return None
        if not isinstance(x, np.ndarray) or x.ndim != 2\
                or not x.size or not np.issubdtype(x.dtype, np.number):
            print("x has to be an numpy.array, a matrix of shape m * n.")
            return None
        if self.thetas.shape[0] != x.shape[1] + 1:
            print("x and theta's shapes don't match.")
            return None
        X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        return self.sigmoid_(X @ self.thetas)

    def loss_elem_(self, y, y_hat, eps=1e-15):
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
                or not y.size or not np.issubdtype(y.dtype, np.number):
            print("y has to be an numpy.array, a vector.")
            return None
        if not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or\
                y_hat.shape[1] != 1 or not y_hat.size or\
                not np.issubdtype(y_hat.dtype, np.number):
            print("y_hat has to be an numpy.array, a vector.")
            return None
        if y.shape[0] != y_hat.shape[0]:
            print('y and y_hat have different shapes')
            return None
        if not isinstance(eps, float):
            print("eps has to be a float.")
            return None
        return y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)

    def loss_(self, y, y_hat, eps=1e-15):
        if not hasattr(self, 'thetas') or not hasattr(self, 'lambda_'):
            return None
        logistic_error = self.loss_elem_(y, y_hat, eps)
        if logistic_error is None:
            return None
        reg = self.thetas[1:, :].T.dot(self.thetas[1:, :]).item()
        m = y.shape[0]
        lambda_ = self.lambda_ if self.penalty == 'l2' else 0
        return float(-logistic_error.sum() / m + lambda_ * reg / (2 * m))
