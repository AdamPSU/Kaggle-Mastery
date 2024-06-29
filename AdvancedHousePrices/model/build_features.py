import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_log_error

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# select features
train = train[['SalePrice', 'OverallQual', 'LotFrontage', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd']]
test_sub = test[['OverallQual', 'LotFrontage', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd']]

# impute NA
train = train.fillna(train.mean())
test_sub = test_sub.fillna(test_sub.mean())

# fitting the model
y = train['SalePrice']
x = train.drop('SalePrice', axis=1)

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(test_sub)
y_pred[y_pred < 0] = 0

test['SalePrice'] = y_pred
submission = test[['Id', 'SalePrice']]

submission.to_csv('../data/AdvancedHousePricesVER1.csv', index=False)




