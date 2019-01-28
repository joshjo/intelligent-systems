
# Linear Regression Multivar

## 1. Reading Data using Panda


```python
import pandas as pd
```


```python
data = pd.read_csv('data/train.csv')
```


```python
data[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>50</td>
      <td>RL</td>
      <td>85.0</td>
      <td>14115</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>700</td>
      <td>10</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>143000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>20</td>
      <td>RL</td>
      <td>75.0</td>
      <td>10084</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>307000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>10382</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Shed</td>
      <td>350</td>
      <td>11</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>200000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>50</td>
      <td>RM</td>
      <td>51.0</td>
      <td>6120</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2008</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>129900</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>190</td>
      <td>RL</td>
      <td>50.0</td>
      <td>7420</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>118000</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 81 columns</p>
</div>




```python
data.shape
```




    (1460, 81)



## 2. Data visualization


```python
import seaborn as sns
```


```python
%matplotlib inline
```


```python
sns.pairplot(data, x_vars='YearRemodAdd', y_vars='SalePrice', height=7, aspect=2, kind='reg')
```

    /Users/josue/.virtualenvs/is-hw1/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval





    <seaborn.axisgrid.PairGrid at 0x117566438>




![png](output_8_2.png)



```python
sns.pairplot(data, x_vars='GrLivArea', y_vars='SalePrice', height=7, aspect=2, kind='reg')
```




    <seaborn.axisgrid.PairGrid at 0x1197a3668>




![png](output_9_1.png)



```python
data['MSZoning'] = data['MSZoning'].map({'A': 1,'C (all)':2, 'FV':3, 'I':4, 'RH':5, 'RL':6, 'RP':7, 'RM':8 })
data['Street'] = data['Street'].map({'Grvl':1 ,'Pave': 2})
data['Alley'] = data['Alley'].map({'NA': 1, 'Grvl':2 ,'Pave': 3})
data['LotShape'] = data['LotShape'].map({'Reg':1, 'IR1':2 ,'IR2':3, 'IR3':4})
data['LandContour'] = data['LandContour'].map({'Lvl': 4,'Bnk': 3, 'HLS': 2, 'Low': 1})
data['Utilities'] = data['Utilities'].map({ 'AllPub':4, 'NoSewr':3, 'NoSeWa' :1, 'ELO': 1})
data['LotConfig'] = data['LotConfig'].map({'Inside': 1, 'Corner': 2, 'CulDSac': 3, 'FR2': 4, 'FR3':5})
data['LandSlope'] = data['LandSlope'].map({'Gtl': 3, 'Mod': 2, 'Sev': 1})
data['Neighborhood'] = data['Neighborhood'].map({ 'Blmngtn': 1, 'Blueste': 2, 'BrDale': 3, 'BrkSide': 4, 'ClearCr': 5, 'CollgCr': 6, 'Crawfor': 7, 'Edwards': 8,'Gilbert': 9, 'IDOTRR': 10, 'MeadowV': 11, 'Mitchel': 12, 'Names': 13, 'NoRidge': 14, 'NPkVill': 15,'NridgHt': 16, 'NWAmes': 17, 'OldTown': 18, 'SWISU': 19, 'Sawyer': 20, 'SawyerW': 21, 'Somerst': 22, 'StoneBr': 23, 'Timber': 24, 'Veenker':25})
data['Condition1'] = data['Condition1'].map({ 'Artery': 1, 'Feedr': 2, 'Norm': 3, 'RRNn': 4, 'RRAn': 5, 'PosN': 6, 'PosA': 7, 'RRNe': 8, 'RRAe':9})
data['Condition2'] = data['Condition2'].map({ 'Artery': 1, 'Feedr': 2, 'Norm': 3, 'RRNn': 4, 'RRAn': 5, 'PosN': 6, 'PosA': 7, 'RRNe': 8, 'RRAe':9})
data['BldgType'] = data['BldgType'].map({'1Fam': 1, '2FmCon': 2, 'Duplx': 3, 'TwnhsE': 4, 'TwnhsI': 5})
data['HouseStyle'] = data['HouseStyle'].map({ '1Story': 1, '1.5Fin': 2, '1.5Unf': 3, '2Story': 4, '2.5Fin': 5, '2.5Unf': 6, 'SFoyer': 7, 'SLvl': 8})
data['RoofStyle'] = data['RoofStyle'].map({'Flat': 1, 'Gable': 2, 'Gambrel': 3, 'Hip': 4, 'Mansard': 5, 'Shed': 6})
data['RoofMatl'] = data['RoofMatl'].map({'ClyTile': 1, 'CompShg': 2, 'Membran': 3, 'Metal': 4, 'Roll': 5, 'Tar&Grv': 6, 'WdShake': 7, 'WdShngl':8})
data['Exterior1st'] = data['Exterior1st'].map({'AsbShng': 1, 'AsphShn': 2, 'BrkComm': 3, 'BrkFace': 4, 'CBlock': 5, 'CemntBd': 6, 'HdBoard': 7, 'ImStucc':8, 'MetalSd': 9 , 'Other': 10, 'Plywood': 11, 'PreCast':12, 'Stone': 13,'Stucco': 14, 'VinylSd': 15, 'Wd Sdng': 16, 'WdShing': 17})
data['Exterior2nd'] = data['Exterior2nd'].map({'AsbShng': 1, 'AsphShn': 2, 'BrkComm': 3, 'BrkFace': 4, 'CBlock': 5, 'CemntBd': 6, 'HdBoard': 7, 'ImStucc':8, 'MetalSd': 9 , 'Other': 10, 'Plywood': 11, 'PreCast':12, 'Stone': 13,'Stucco': 14, 'VinylSd': 15, 'Wd Sdng': 16, 'WdShing': 17})
data['MasVnrType'] = data['MasVnrType'].map({'BrkCmn': 1, 'BrkCmn': 2, 'CBlock': 3, 'None': 4, 'Stone': 5})
data['ExterQual'] = data['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
data['ExterCond'] = data['ExterCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
data['Foundation'] = data['Foundation'].map({'BrkTil': 1, 'CBlock': 2, 'PConc': 3, 'Slab': 4, 'Stone': 5, 'Wood': 6})
data['BsmtQual'] = data['BsmtQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data['BsmtCond'] = data['BsmtCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data['BsmtExposure'] = data['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
data['BsmtFinType1'] = data['BsmtFinType1'].map({'NA': 0,'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
data['BsmtFinType2'] = data['BsmtFinType2'].map({'NA': 0,'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
data['Heating'] = data['Heating'].map({'Floor': 1, 'GasA': 2, 'GasW': 3, 'Grav': 4, 'OthW': 5, 'Wall': 6})
data['HeatingQC'] = data['HeatingQC'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
data['CentralAir'] = data['CentralAir'].map({'N': 0, 'Y': 1})
data['Electrical'] = data['Electrical'].map({'SBrkr': 1, 'FuseA': 2, 'FuseF': 3, 'FuseP': 4, 'Mix': 5})
data['KitchenQual'] = data['KitchenQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
data['Functional'] = data['Functional'].map({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ':8})
data['FireplaceQu'] = data['FireplaceQu'].map({'TA': 1, 'Gd': 2, 'Ex': 3})
data['GarageType'] = data['GarageType'].map({'NA': 0, 'Detchd': 1, 'CarPort': 2, 'BuiltIn': 3, 'Basment': 4, 'Attchd': 5, '2Types': 6})
data['GarageFinish'] = data['GarageFinish'].map({'NA': 0,'Unf': 1, 'RFn': 2, 'Fin': 3})
data['GarageQual'] = data['GarageQual'].map({'NA':0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
data['GarageCond'] = data['GarageCond'].map({'NA':0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
data['PavedDrive'] = data['PavedDrive'].map({'N': 1, 'P': 2, 'Y': 3})
data['PoolQC'] = data['PoolQC'].map({'NA':0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
data['Fence'] = data['Fence'].map({'NA':0,'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4})
data['MiscFeature'] = data['MiscFeature'].map({'NA':0,'TenC': 1, 'Shed': 2, 'Othr': 3, 'Gar2': 4, 'Elev': 5})
data['SaleType'] = data['SaleType'].map({'WD': 1, 'CWD': 2, 'VWD': 3, 'New': 4, 'COD': 5, 'Con': 6, 'ConLw': 7, 'ConLI':8, 'ConLD': 9, 'Oth': 10 })
data['SaleCondition'] = data['SaleCondition'].map({'Normal': 1, 'Abnorml': 2, 'AdjLand': 3, 'Alloca': 4, 'Family': 5, 'Partial': 6})
```


```python
sns.pairplot(data, x_vars='BsmtFinSF1', y_vars='SalePrice', height=7, aspect=2, kind='reg')
```




    <seaborn.axisgrid.PairGrid at 0x11b3f9550>




![png](output_11_1.png)



```python
sns.pairplot(data, x_vars='1stFlrSF', y_vars='SalePrice', height=7, aspect=2, kind='reg')
```




    <seaborn.axisgrid.PairGrid at 0x11b440da0>




![png](output_12_1.png)



```python
sns.pairplot(data, x_vars='Alley', y_vars='SalePrice', height=7, aspect=2, kind='reg')
```




    <seaborn.axisgrid.PairGrid at 0x11bc8a9e8>




![png](output_13_1.png)



```python
sns.pairplot(data, x_vars='LotShape', y_vars='SalePrice', height=7, aspect=2, kind='reg')
```




    <seaborn.axisgrid.PairGrid at 0x11c0be470>




![png](output_14_1.png)



```python
sns.pairplot(data, x_vars='LandContour', y_vars='SalePrice', height=7, aspect=2, kind='reg')
```




    <seaborn.axisgrid.PairGrid at 0x11c30a3c8>




![png](output_15_1.png)



```python
sns.pairplot(data, x_vars=data.columns, y_vars='SalePrice', height=7, aspect=0.5, kind='reg')
```




    <seaborn.axisgrid.PairGrid at 0x11dec5550>




![png](output_16_1.png)


## 3 Computing linear regression with gradient descend linear regression [own implementation]



```python
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

```

    cost 1.8155652800713652e+25


## 4 Using linear regresion with sklearn


```python
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

```
    score: 1.0
    predictions [172500 172500 172500 ... 172500 141000 172500]



```python

```

## Kaggle result

![png](kaggle_result.png)
