import numpy as np
import pandas as pd
import math
from map_data import map_data

df = pd.read_csv('data/train.csv', index_col=0)

columns = [
    'MSZoning',
    'OverallQual',
    'ExterQual',
    'BsmtQual',
    'KitchenQual',
    'Alley',
    'LotShape',
    'LandContour',
    'TotalBsmtSF',
    'Utilities',
    'LotConfig',
    'Neighborhood',
    'YearRemodAdd'
]

data = map_data(df)

data = df[columns].values

Y = df['SalePrice'].values

M = len(data)
N = len(data[0])

data = np.nan_to_num(data)

X = np.ones((M, N + 1))

X[:,1:] = data


T = np.zeros(N + 1)


def HT(T, x_row):
    return sum([t * x for t, x in zip(T, x_row)])


def cost_function():
    J_t = 0

    for x_row, y in zip(X, Y):
        h_t = HT(T, x_row)
        J_t += (h_t - y)**2

    return J_t / (2*M)


def regression(alpha=0.01, err=0.00000001):
    current_error = math.inf
    prev_cost = 0
    num_iters = 0
    for _ in range(1):
        num_iters += 1
        for i, _ in enumerate(T):
            sum_T_i = 0
            for j, (x_row, y) in enumerate(zip(X, Y)):
                h_t = HT(T, x_row)
                sum_T_i += (h_t - y) * X[j][i]
            T[i] -= (alpha * sum_T_i / M)
        cost = cost_function()
        current_error = abs(prev_cost - cost)
        prev_cost = cost
    return(T, prev_cost, num_iters)


dt = pd.read_csv('data/test.csv', index_col=0)
test = map_data(dt)


test = dt[columns].values
test = np.nan_to_num(test)

xtest = np.ones((len(test), N + 1))
xtest[:,1:] = test

if __name__ == '__main__':
    T, cost, num_iters = regression()
    print('cost', cost)
    y = []
    for row in xtest:
        y.append(sum([t * x for t, x in zip(T, row)]))
    result = pd.DataFrame({'SalePrice': y}, index=dt.index)
    result.to_csv('data/result.csv')
