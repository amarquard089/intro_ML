import numpy as np

class Perceptron:
    """
    Perceptron classifier
    nue: learning rate
    n_iter: iterations
    """
    def __init__(self, nue = 0.01, n_iter = 10):
        self.nue = nue
        self.n_iter = n_iter


    def fit(self, X, y):
        """
        Fit the training data
        """
        # dimension
        p = X.shape[0]
        p2 = p + 1

        # weights initialization
        self.w_ = np.zeros(p2)

        # error term
        self.errors_ = []

        # preceptron algorithm
        for _ in range(self.n_iter):
            # in each step we calculate the weights and update them
            # for that we use the formula w = nü * (y - y_pred) * x
            # whereas w[1:] = nü * (y - y_pred) * x
            # w[0] = nü * (y - y_pred)
            error = 0
            for i in range(p):
                # calculate z
                z = X[i] @ self.w_[1:] + self.w_[0]
                # predict y
                y_pred = 1 if z >= 0 else -1
                update = self.nue * (y[i] - y_pred)
                self.w_[1:] += update * X[i]
                self.w_[0] += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self

    def calculate_z(self, X):
        """
        Calculate net input
        """
        return X @ self.w_[1:] + self.w_[0]
    
    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def __str__(self):
        return 

    def __repr__(self):
        return "Perceptron(nue = %.2f, n_iter = %d)" % (self.nue, self.n_iter)