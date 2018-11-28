import pandas as pd

from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from pandas import DataFrame


def train(x_train, y_train, epochs=8000, alpha=0.0001):
    """
    Function which trains an SVM model
    """
    dim = len(x_train[0])
    w = np.zeros(dim)
    for epoch in range(1, epochs + 1):
        y = np.array([[sum(i)] for i in (w * x_train)])
        pp = y * y_train
        count = 0
        for j, val in enumerate(pp):
            if(val >= 1):
                cost = 0
                w = w - alpha * (2 * 1 / epochs * w)
            else:
                cost = 1 - val
                w = w + alpha * (x_train[j] * y_train[j] - 2 * 1 / epoch * w)
    return w


def get_accuracy(x_test, weights):
    y_pred = weights * x_test
    y_pred = [sum(i) for i in y_pred]
    predictions = []
    for val in y_pred:
        if(val > 1):
            predictions.append(1)
        else:
            predictions.append(-1)
    return accuracy_score(y_test,predictions)


if __name__ == '__main__':
    X, yy = make_blobs(n_samples=100, centers=2, n_features=2)
    Y = np.array([[i if i > 0 else -1] for i in yy])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)
    w = train(x_train, y_train)
    acc = get_accuracy(x_test, w)
    plot_data(X, yy, w)
    print('accuracy', acc)
