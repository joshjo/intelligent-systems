import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame

# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2)


# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()



# train_f1 = x_train[:,0]
# train_f2 = x_train[:,1]

# train_f1 = train_f1.reshape(90,1)
# train_f2 = train_f2.reshape(90,1)

# w1 = np.zeros((90,1))
# w2 = np.zeros((90,1))

# epochs = 1
# alpha = 0.0001

# while(epochs < 10000):
#     y = w1 * train_f1 + w2 * train_f2
#     prod = y * y_train
#     print(epochs)
#     count = 0
#     for val in prod:
#         if(val >= 1):
#             cost = 0
#             w1 = w1 - alpha * (2 * 1/epochs * w1)
#             w2 = w2 - alpha * (2 * 1/epochs * w2)

#         else:
#             cost = 1 - val
#             w1 = w1 + alpha * (train_f1[count] * y_train[count] - 2 * 1/epochs * w1)
#             w2 = w2 + alpha * (train_f2[count] * y_train[count] - 2 * 1/epochs * w2)
#         count += 1
#     epochs += 1
