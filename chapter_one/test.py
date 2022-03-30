import numpy as np
import os
import pandas as pd

from perceptron import Perceptron


if __name__ == "__main__":
    a = os.path.join('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    df = pd.read_csv(a, header=None, encoding='utf-8')
    print(df.head())
    
    a = Perceptron(nue = 0.01, n_iter = 10)
    a.fit(np.array([[1,2],[3,4]]), np.array([1,-1]))
    a.errors_
    a.w_
