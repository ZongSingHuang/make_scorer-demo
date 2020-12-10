# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 08:07:51 2020

@author: ZongSing_NB
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer

def mape(y_true, y_pred):
    # 一定是左實際值，右預測值
    print('target is [{target}]'.format(target=y_true))
    print('predict is [{predict}]'.format(predict=y_pred))
    error = np.abs( (y_true-y_pred)/y_true )
    return np.mean(error)

# y = 2 + 0*x_1 + 0*x_2
aaa = np.ones(10)
aaa = np.vstack([aaa, aaa]).T
bbb = np.ones(10)*2
model = LinearRegression()
model.fit(aaa, bbb)

my_loss = make_scorer(mape)

# (模型, 輸入, 目標)
print('loss is [{mape}]'.format(mape=my_loss(model, aaa, bbb)))

