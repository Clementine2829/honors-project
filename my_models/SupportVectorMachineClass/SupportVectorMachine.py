import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        print("Features")
        print(n_features)
        self.w = np.zeros(n_features)
        self.b = 0

    def _satisfy_constraint(self, x, idx):
        linear_model = np.dot(x, self.w) + self.b
        return self.cls_map[idx] * linear_model >= 1

    def _get_gradients(self, constrain, x, idx):
        # if data point lies on the correct side
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db
        # if data point is on the wrong side
        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db

    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def fit(self, X, y):
        # init weights & biases
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0
        # map binary class to {-1, 1}
        self.cls_map = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                # check if data point satisfies the constraint
                constrain = self._satisfy_constraint(x, idx)
                # compute the gradients accordingly
                dw, db = self._get_gradients(constrain, x, idx)
                # update weights & biases
                self._update_weights_bias(dw, db)

    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        # compute the sign
        prediction = np.sign(estimate)
        # map class from {-1, 1} to original values {0, 1}
        return np.where(prediction == -1, 0, 1)
