import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


df = pd.read_csv('iris.csv')
df = df.drop(['Id'],axis=1)
target = df['Species']
s = set()
for val in target:
    s.add(val)
s = list(s)
rows = list(range(100,150))
df = df.drop(df.index[rows])


x = df['SepalLengthCm']
y = df['PetalLengthCm']

setosa_x = x[:50]
setosa_y = y[:50]

versicolor_x = x[50:]
versicolor_y = y[50:]

plt.figure(figsize=(8,6))
plt.scatter(setosa_x,setosa_y,marker='+',color='green')
plt.scatter(versicolor_x,versicolor_y,marker='_',color='red')


## Drop rest of the features and extract the target values
df = df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)
Y = []
target = df['Species']
for val in target:
    if(val == 'Iris-setosa'):
        Y.append(-1)
    else:
        Y.append(1)
df = df.drop(['Species'],axis=1)
X = df.values.tolist()
## Shuffle and split the data into training and test set
X, Y = shuffle(X,Y)
x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)

## Support Vector Machine

train_f1 = x_train[:,0]
train_f2 = x_train[:,1]

train_f1 = train_f1.reshape(len(y_train), 1)
train_f2 = train_f2.reshape(len(y_train), 1)

# w1 = np.zeros((90,1))
# w2 = np.zeros((90,1))
dim = 2
w = np.zeros(dim)

epochs = 5000
alpha = 0.0001

# print(x_train)

for epoch in range(1, epochs + 1):
    y = np.array([[sum(i)] for i in (w * x_train)])
    pp = y * y_train
    # print('pp', y)
    count = 0

    for j, val in enumerate(pp):
        # print('val', val)
        if(val >= 1):
            cost = 0
            w = w - alpha * (2 * 1 / epochs * w)
        else:
            cost = 1 - val
            w = w + alpha * (x_train[j] * y_train[j] - 2 * 1 / epoch * w)


# print('weights', w)


# # Clip the weights
# index = list(range(10,90))
# w1 = np.delete(w1,index)
# w2 = np.delete(w2,index)

# w1 = w1.reshape(10,1)
# w2 = w2.reshape(10,1)


# print('y_train', y_train)


## Extract the test data features
# test_f1 = x_test[:,0]
# test_f2 = x_test[:,1]

# test_f1 = test_f1.reshape(10,1)
# test_f2 = test_f2.reshape(10,1)

## Predict
y_pred = w * x_test
y_pred = [sum(i) for i in y_pred]
predictions = []
for val in y_pred:
    if(val > 1):
        predictions.append(1)
    else:
        predictions.append(-1)

print(accuracy_score(y_test,predictions))
