from sklearn import linear_model

import numpy as np
from map_data import map_data
import pandas as pd
import warnings


warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

df = pd.read_csv('data/train.csv', index_col=0)

columns = [
    'GrLivArea',
    'YearBuilt',
    'MSSubClass',
    'LotArea',
    'YearRemodAdd',
    'MasVnrArea',
    'Foundation',
    'BsmtFinSF1',
    'TotalBsmtSF',
    '1stFlrSF',
    'TotRmsAbvGrd',
    'GarageYrBlt',
    'GarageArea',
]

# X = np.nan_to_num(map_data(df[columns]).values)
X = np.nan_to_num(map_data(df)[columns].values)

y = df['SalePrice'].values
lm = linear_model.SGDClassifier()
model = lm.fit(X, y)

dt = pd.read_csv('data/test.csv', index_col=0)
dt = map_data(dt)
Xt = np.nan_to_num(map_data(dt)[columns].values)

predictions = lm.predict(Xt)
print('score:', lm.score(Xt, predictions))
print('predictions', predictions)
result = pd.DataFrame({'SalePrice': predictions}, index=dt.index)
result.to_csv('data/result_custom.csv')
