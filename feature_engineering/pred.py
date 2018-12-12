# -*- coding: utf-8 -*-
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn.preprocessing import scale, StandardScaler
import matplotlib.pyplot as plt

# training set
df_train = pd.read_csv(r"../data/train1.csv")
train_dev_x = df_train.iloc[:, 1: -1]
train_dev_y = df_train.iloc[:, -1]
train_dev_columns = train_dev_x.columns

# test set
test_x = pd.read_csv(r"../data/test1.csv").iloc[:, 1:]
train_dev_x, test_x = train_dev_x.align(test_x, join="left", axis=1)
test_columns = test_x.columns
# check the alignment between the features of training set and test set
# print([c for c in train_dev_columns.values if c not in test_columns.values])
# print(train_dev_x)
# print(test_x)

# features scale
scaler = StandardScaler().fit(train_dev_x.values)
train_dev_x = scaler.transform(train_dev_x.values)
test_x = scaler.transform(test_x.values)

# label log
train_dev_y = np.log(train_dev_y)

# training
'''
kf = KFold(n_splits=10, random_state=None, shuffle=False)
for train_index, dev_index in kf.split(train_dev_x):
    train_x, dev_x = train_dev_x[train_index], train_dev_x[dev_index]
    train_y, dev_y = train_dev_y[train_index], train_dev_y[dev_index]

    # try the ridge regression
    ridge = Ridge().fit(train_x, train_y)
    dev_pred_y = ridge.predict(dev_x)
    print(np.sqrt(mean_squared_error(dev_y, dev_pred_y)))
    # print(dev_y[: 10])
    # print(dev_pred_y[: 10])
'''
alphas = np.logspace(-3, 3, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, train_dev_x, train_dev_y, cv=10, scoring="neg_mean_squared_error"))
    test_scores.append(np.mean(test_score))

plt.plot(alphas, test_scores)
plt.show()





