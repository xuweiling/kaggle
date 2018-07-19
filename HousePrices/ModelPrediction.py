
# 回归里面主要包括：SGD(随机梯度下降回归)，lasso，岭回归。
# 接下来我们就先用岭回归跑一跑，之所以选择岭回归是因为这种回归在多特征情
# 况下时，我们可以直接把所有特征放进去建模，不需要考虑特征选取。
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score #用交叉验证来测试模型
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VS_CV_ERROR = 'max_depth vs CV Error'

df_train = pd.read_csv('./data/handled/train_outlier_dummies.csv')
df_test = pd.read_csv('./data/handled/test_outlier_dummies.csv')
#训练数据和测试数据列数不一样
df_test = df_test.ix[:, df_train.columns].fillna(0)
# print(df_train.info())
# print('****************************************')
# print(df_test.info())



#检查是否有空值
# print(np.isnan(df_train).any())

#这一步是在把DataFrame转换成Numpy Array格式，这一步不是必要的，只是便于之后的工作
X_train = df_train.drop('SalePrice', 1, inplace=False).values
y_train = df_train['SalePrice'].values
X_test = df_test.drop('SalePrice', 1, inplace=False).values



# 接下来使用交叉验证去找到模型最佳参数.
# 简单来说，交叉验证的好处就是考虑了多种可能之后，
# 尽全力找到你的数据最好的划分方式，用这种方式算出最好的参数。
alphas = np.logspace(-3, 2, 50)  #logspace用于创建等比数列，本例从10^-3到10^2中选50个数
test_scores = []#这里面会装每一次交叉验证的得分，最后我们可以找到最好的参数，这就是调参
# for alpha in alphas:
#     clf = Ridge(alpha)  #clf是分类器
#     #这里用的10-fold Cross Validation（
#     # 就是十折交叉验证，用来测试精度。
#     # 是常用的精度测试方法。将数据集分成十分，轮流将其中9份做训练1份做测试，10次的结果的均值作为对算法精度的估计，一般还需要进行多次10倍交叉验证求均值，例如10次10倍交叉验证，更精确一点。）
#     # np.sqrt()函数用来给一个列表中每一个元素求根号
#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
#
# #可视化参数和分数之间的关系
# plt.plot(alphas, test_scores)
# plt.title('Ridge Alpha vs Error')
# plt.show()
#因为我们的评分方法是去看误差，所以肯定是分数越低效果越好，从图中可以看出，参数在0-10左右时，误差来到了一个最小值

#在这之后我们来看看RandomForest(随机森林)
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
# for max_feat in max_features:
#     clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
# plt.plot(max_features, test_scores)
# plt.title('RandomForest Alpha vs Error')
# plt.show()
#从图中可以看出，参数在0.4-0.6之间时，误差值最小

# 这样，我们就通过sklearn简单的使用了两个模型了，但是这样在竞赛中正确率往往是不够的，
# 一般的解决方式是Ensemble(集成学习)。这里面主要有Bagging,Boosting以及Stacking等。

#先介绍stacking的思维（简单来说就是汲取多种模型的优点）
#先把我们用到的两个模型最好的参数输入模型

ridge = Ridge(alpha=5)
rf = RandomForestRegressor(n_estimators=500, max_features=.5)
#进行训练
ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)

#已经可以预测了，注意要做log运算的反运算
y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))
#然后进行两个模型的融合（这里就是求个平均数）
y_final = (y_ridge + y_rf) / 2
# print(y_final)

#尝试用Ensemble进行优化
#这里先定义一个基本的弱分类器（之前的岭回归）
ridge = Ridge(alpha=5)
#准备用Bagging
#使用Bagging时，在base_estimator里填写使用的弱分类器
# params = [1, 10, 15, 20, 25, 30, 40]
# test_scores = []
# for param in params:
#     clf = BaggingRegressor(n_estimators=param, base_estimator=ridge)
#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
# plt.plot(params, test_scores)
# plt.title('n_estimator vs CV Error')
# plt.show()

#用XgBoost进行优化
# params = [1,2,3,4,5,6]
# test_score = []
# for param in params:
#     clf = XGBRegressor(max_depth=param)
#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
# plt.plot(params, test_scores)
# plt.title('max_depth vs CV Error' % VS_CV_ERROR)
# plt.show()

xgbRegressor = XGBRegressor(4)
#进行训练
xgbRegressor.fit(X_train, y_train)
df_test['SalePrice'] = np.expm1(xgbRegressor.predict(X_test))
# print(df_test['SalePrice'])
df_test.to_csv('./data/handled/submission_xgboosting.csv', columns=['Id', 'SalePrice'], index=False)