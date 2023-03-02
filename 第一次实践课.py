import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
import sklearn.datasets
from pprint import pprint
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import warnings

# def not_empty(s):
#     return s != ''
#
# if __name__ == "__main__":
#
#     file_data = pd.read_csv('housing.data', header=None)
#
#     data = np.empty((len(file_data), 14))
#     for i, d in enumerate(file_data.values):
#         d = list(map(float, list(filter(not_empty, d[0].split(' ')))))
#         data[i] = d
#     #可修改特征
#     x, y = np.split(data, (13, ), axis=1)
#
#     print('样本个数：%d, 特征个数：%d' % x.shape)
#     print(y.shape)
#     y = y.ravel()
#
#     x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
#     #可修改 模型
#     model = LinearRegression()
#
#     print('开始建模...')
#     model.fit(x_train, y_train)
#
#
#     order = y_test.argsort(axis=0)
#     y_test = y_test[order]
#     x_test = x_test[order, :]
#     y_pred = model.predict(x_test)
#     r2 = model.score(x_test, y_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print('R2:', r2)
#     print('均方误差：', mse)
#
#     t = np.arange(len(y_pred))
#     mpl.rcParams['font.sans-serif'] = ['simHei']
#     mpl.rcParams['axes.unicode_minus'] = False
#     plt.figure(facecolor='w')
#     plt.plot(t, y_test, 'r-', lw=2, label='真实值')
#     plt.plot(t, y_pred, 'g-', lw=2, label='估计值')
#     plt.legend(loc='best')
#     plt.title('波士顿房价预测', fontsize=18)
#     plt.xlabel('样本编号', fontsize=15)
#     plt.ylabel('房屋价格', fontsize=15)
#     plt.grid()
#     plt.show()


def not_empty(s):
    return s != ''

if __name__ == "__main__":

    file_data = pd.read_csv('housing.data', header=None)

    data = np.empty((len(file_data), 14))
    for i, d in enumerate(file_data.values):
        d = list(map(float, list(filter(not_empty, d[0].split(' ')))))
        data[i] = d
    #去掉存在空值的行
    data = data[~np.isnan(data).any(axis=1)]
    #去掉异常值（假设异常值为大于30的MEDV）
    data = data[data[:, 13] < 30]
    #可修改特征
    x, y = np.split(data, (13, ), axis=1)

    print('样本个数：%d, 特征个数：%d' % x.shape)
    print(y.shape)
    y = y.ravel()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    #可修改 模型
    model = LinearRegression()

    print('开始建模...')
    model.fit(x_train, y_train)


    order = y_test.argsort(axis=0)
    y_test = y_test[order]
    x_test = x_test[order, :]
    y_pred = model.predict(x_test)
    r2 = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print('R2:', r2)
    print('均方误差：', mse)

    t = np.arange(len(y_pred))
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', lw=2, label='真实值')
    plt.plot(t, y_pred, 'g-', lw=2, label='估计值')
    plt.legend(loc='best')
    plt.title('波士顿房价预测', fontsize=18)
    plt.xlabel('样本编号', fontsize=15)
    plt.ylabel('房屋价格', fontsize=15)
    plt.grid()
    plt.show()