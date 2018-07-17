import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#读数据
df_train = pd.read_csv('./data/train.csv')
#查看列信息
# print(df_train.columns)
# Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
#        'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
#        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
#        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
#        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
#        'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
#        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
#        'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
#        'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
#        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
#        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
#        'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
#        'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
#        'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
#        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
#        'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
#        'SaleCondition', 'SalePrice'],
#       dtype='object')

#房价描述性数据
# print(df_train['SalePrice'].describe())
# count      1460.000000
# mean     180921.195890
# std       79442.502883
# min       34900.000000
# 25%      129975.000000
# 50%      163000.000000
# 75%      214000.000000
# max      755000.000000
# Name: SalePrice, dtype: float64

# 绘制直方图
# sns.distplot(df_train['SalePrice'])
# sns.plt.show()
#从直方图中看出，房价偏离正太分布；数据正偏（右侧偏长）；有峰值

#输出数据的偏度和峰度
# print("Skewness: %f" % df_train['SalePrice'].skew())#偏度 1.882876
# print("Kurtosis: %f" % df_train['SalePrice'].kurt())#峰度 6.536282

#房价的相关变量分析: 选出自己认为跟房价相关性较高的几个变量
#与数据型变量的关系
numCol1 = 'LotArea'    #地块面积
numCol2 = 'GrLivArea' #以上（地面）生活区平方英尺


# print(df_train[numCol1].describe())
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.scatter(df_train[numCol1], df_train['SalePrice'])
# plt.show()  #可以看出SalePrice和LotArea关系比较密切，基本呈线性关系

# ax1.scatter(df_train[numCol2], df_train['SalePrice'])
# plt.show()  #可以看出SalePrice和GrLivArea关系很密切，基本呈线性关系


#与类别型变量的关系
catCol1 = 'MSSubClass'  #识别销售涉及的住宅类型。
catCol2 = 'MSZoning'   #标识销售的一般分区分类
catCol3 = 'Neighborhood'   #周边建筑
catCol4 = 'Condition1'   #周边交通
catCol5 = 'Condition2'   #多个周边交通
catCol6 = 'BldgType'   #住宅类型
catCol7 = 'HouseStyle'   #住宅风格
catCol8 = 'BedroomAbvGr'  #楼层以上的卧室（不包括地下室卧室）
catCol9 = 'TotRmsAbvGrd'  #房间总数（不包括浴室）
catCol10 = 'YearBuilt'  #建造年份
catCol11 = 'OverallQual'

# data = pd.concat([df_train['SalePrice'], df_train[catCol1]], axis=1)
# f, ax = plt.subplots(figsize=(8,6))
# fig = sns.boxplot(x=catCol1, y='SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# sns.plt.show()    # 可以看出MSSubClass取值为60、120、20、75时SalePrice比较高

# data = pd.concat([df_train['SalePrice'], df_train[catCol2]], axis=1)
# f, ax = plt.subplots(figsize=(8,6))
# fig = sns.boxplot(x=catCol2, y='SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# sns.plt.show()  #MSZoning为FV、RL售价较高，MSZoning为C售价较低，MSZoning为A、I、RP的类型没有售卖数据

# data = pd.concat([df_train['SalePrice'], df_train[catCol3]], axis=1)
# f, ax = plt.subplots(figsize=(20,6))
# fig = sns.boxplot(x=catCol3, y='SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# sns.plt.show()  #Neighborhood为NridgHT、NoRidge、StoneBr时，SalePrice较高

# data = pd.concat([df_train['SalePrice'], df_train[catCol4]], axis=1)
# f, ax = plt.subplots(figsize=(10,6))
# fig = sns.boxplot(x=catCol4, y='SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# sns.plt.show()  #Condition1为RRNn, PosN, Norm, PosA时,SalePrice较高

# data = pd.concat([df_train['SalePrice'], df_train[catCol5]], axis=1)
# f, ax = plt.subplots(figsize=(10,6))
# fig = sns.boxplot(x=catCol5, y='SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# sns.plt.show()  #Condition2为PosN, Norm, PosA时SalePrice比较高

# data = pd.concat([df_train['SalePrice'], df_train[catCol6]], axis=1)
# f, ax = plt.subplots(figsize=(10,6))
# fig = sns.boxplot(x=catCol6, y='SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# sns.plt.show()   #BldgType为1Fam, TwnhsE时SalePrice比较高,其他类别相差不大

# data = pd.concat([df_train['SalePrice'], df_train[catCol7]], axis=1)
# f, ax = plt.subplots(figsize=(10,6))
# fig = sns.boxplot(x=catCol7, y='SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# sns.plt.show()  #HouseStyle为2Story, 2.5Fin, 1Story时,房价较高

# data = pd.concat([df_train['SalePrice'], df_train[catCol8]], axis=1)
# f, ax = plt.subplots(figsize=(10,6))
# fig = sns.boxplot(x=catCol8, y='SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# sns.plt.show()  #BedroomAbvGr为0,4,5时房价较高

# data = pd.concat([df_train['SalePrice'], df_train[catCol9]], axis=1)
# f, ax = plt.subplots(figsize=(10,6))
# fig = sns.boxplot(x=catCol9, y='SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# sns.plt.show()  #TotRmsAbvGrd个数为2-11,房价上升,TotRmsAbvGrd>11时房价下降,两个变量之间有较强的关系

# data = pd.concat([df_train['SalePrice'], df_train[catCol10]], axis=1)
# f, ax = plt.subplots(figsize=(20,6))
# fig = sns.boxplot(x=catCol10, y='SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# sns.plt.show()  #YearBuilt和房价之间关系没有很强的趋势性,但是可以看出建筑时间较近的房屋价格更高

# data = pd.concat([df_train['SalePrice'], df_train[catCol11]], axis=1)
# f, ax = plt.subplots(figsize=(20,6))
# fig = sns.boxplot(x=catCol11, y='SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# sns.plt.show()  #OverallQual越高房价越高

#数值型变量:LotArea, GrLivArea与SalePrice似乎线性相关,并且都是正相关.LotArea斜率更高一些
#类别型变量:TotRmsAbvGrd, OverallQual, YearBuilt和房价有关,TotRmsAbvGrd, OverallQual的相关性更强,
#TotRmsAbvGrd值在11附近房价最高,OverallQual越高房价越高. 其他的类别没有趋势性,但是存在不同类别对应房价不同的情况

#客观分析
#相关系数矩阵  (计算所有特征值每两个之间的相关系数，并作图表示。)
# corrmat = df_train.corr()
# f, ax = plt.subplots(figsize=(12, 8))
# sns.heatmap(corrmat, vmax=.8, square=True)
# sns.set(font_scale=1)
# pl.xticks(rotation=90)
# pl.yticks(rotation=360)
# sns.plt.show()
#图中不同变量之间，颜色越强的相关性越强，可以看出相关系数最大的有，
# TotalBsmtSF和1stFlrSF； Garage变量群； YearBuilt和GarageYearBuilt;
#和SalePrice相关性较大的有（相关性依次递减）：
# OverallQual; GrLivArea; TotalBsmtSF、lstFlrSF、GarageCars、GarageArea


#SalePrice相关系数矩阵
k=10  #(取出相关性最大的前十个，做出热点图表示)
corrmat = df_train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(df_train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
#                  annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
# pl.xticks(rotation=90)
# pl.yticks(rotation=360)
# plt.show()
#从图中可以看出
#1、OverallQual, GrLivArea 以及 TotalBsmtSF 与 SalePrice有很强的相关性。
#2、 GarageCars 和 GarageArea 也是相关性比较强的变量.
# 车库中存储的车的数量是由车库的面积决定的，
# 不需要专门区分GarageCars 和 GarageArea ，所以我们只需要其中的一个变量。
# 这里我们选择了GarageCars因为它与SalePrice的相关性更高一些。
# 3、TotalBsmtSF 和 1stFloor 与上述情况相同，选择 TotalBsmtSF。
# 4、FullBath几乎不需要考虑。
# 5、TotRmsAbvGrd和 GrLivArea与2情况相同，选择，GrLivArea。
# 6、YearBuilt 和 SalePrice相关性似乎不强。


#SalePrice和相关变量之间的散点图
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols], size=2.5)
# plt.show()
# 尽管我们已经知道了一些主要特征，这一丰富的散点图给了我们一个关于变量关系的合理想法。
#1、TotalBsmtSF 和 GrLiveArea之间的散点图是很有意思的。
# 我们可以看出这幅图中，一些点组成了线，就像边界一样。
# 大部分点都分布在那条线下面，这也是可以解释的。地下室面积和地上居住面积可以相等，
# 但是一般情况下不会希望有一个比地上居住面积还大的地下室。
# 2、SalePrice 和YearBuilt 之间的散点图也值得我们思考。
# 在“点云”的底部，我们可以观察到一个几乎呈指数函数的分布。
# 我们也可以看到“点云”的上端也基本呈同样的分布趋势。
# 并且可以注意到，近几年的点有超过这个上端的趋势。


#缺失数据
# 关于缺失数据需要思考的重要问题：
# 1、这一缺失数据的普遍性如何？
# 2、缺失数据是随机的还是有律可循？
# 这些问题的答案是很重要的，因为缺失数据意味着样本大小的缩减，这会阻止分析进程。
# 除此之外，以实质性的角度来说，需要保证对缺失数据的处理不会出现
# 偏离或隐藏任何难以忽视的真相。
total = df_train.isnull().sum().sort_values(ascending=False)   #df.isnull().sum()返回每列包含的缺失值的个数
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)  #df_train.isnull().count()df中所有值的个数
missing_data = pd.concat([total,percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(20))
# 1、当超过15%的数据都缺失的时候，我们应该删掉相关变量且假设该变量并不存在。
# 根据这一条，一系列变量都应该删掉，例如PoolQC, MiscFeature, Alley等等，
# 这些变量都不是很重要，因为他们基本都不是我们买房子时会考虑的因素。
# 2、Garage 变量群的缺失数据量都相同，由于关于车库的最重要的信息都可以由GarageCars 表达，
# 并且这些数据只占缺失数据的5%，我们也会删除上述的Garage X  变量群。
# 3、同样的逻辑也适用于 Bsmt X 变量群。（用BsmtFinSF1）
# 4、对于 MasVnrArea 和 MasVnrType，我们可以认为这些因素并不重要。
# 除此之外，他们和YearBuilt以及 OverallQual都有很强的关联性，
# 而这两个变量我们已经考虑过了。所以删除 MasVnrArea和 MasVnrType并不会丢失信息。
# 5、最后，由于Electrical中只有一个损失的观察值，所以我们删除这个观察值，但是保留这一变量。
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)#删除整列
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
# print(df_train.isnull().sum().max())  #检查是否还有缺失值
# df_train.to_csv('./data/handled/train_misingvalue_handled.csv', index=False, encoding='utf-8')


#异常值

#单因素分析
# 这里的关键在于如何建立阈值，定义一个观察值为异常值。我们对数据进行正态化，意味着把数据值转换成均值为0，方差为1的数据。
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])#np.newaxis的功能是插入新维度。
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
# print('outer range(low) of the distribution:', low_range)
# print('outer range (high) of the distribution:', high_range)
# 进行正态化后，可以看出：
# 低范围的值都比较相似并且在0附近分布。
# 高范围的值离0很远，并且七点几的值远在正常范围之外。

#双变量分析
#1、GrLivArea和SalePrice双变量分析
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# plt.show()
# 从图中可以看出：
# 有两个离群的’GrLivArea’ 值很高的数据，我们可以推测出现这种情况的原因。
# 或许他们代表了农业地区，也就解释了低价。
# 这两个点很明显不能代表典型样例，所以我们将它们定义为异常值并删除。
# 图中顶部的两个点是七点几的观测值，他们虽然看起来像特殊情况，
# 但是他们依然符合整体趋势，所以我们将其保留下来。

#删除点
df_train.sort_values(by=var, ascending=False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#2、TotalBsmtSF和SalePrice双变量分析
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# plt.show()
# 从图中可以看出：
# TotalBsmtSF(地下室总面积)和房价有成正比的趋势，在地下室总面积变大的时候，
#散点图有变分散，但是整体还是符合正比的关系的。

#核心部分
# 我们已经进行了数据清洗，并且发现了“SalePrice”的很多信息，
# 现在我们要更进一步理解‘SalePrice’如何遵循统计假设，可以让我们应用多元技术。
# 应该测量4个假设量：
# 正态性
# 同方差性
# 线性
# 相关错误缺失

#正态性
# 应主要关注以下两点：
# 直方图 – 峰度和偏度。
# 正态概率图 – 数据分布应紧密跟随代表正态分布的对角线。
#1、SalePrice
#绘制直方图和正太概率图
# sns.distplot(df_train['SalePrice'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'], plot=plt)
# plt.show()
# 可以看出，房价分布不是正态的，显示了峰值，正偏度，但是并不跟随对角线。
#数据的正太概率图在最大值和最小值有点偏离直线，所以这个属性的左右两边尾部的数据要多于高斯分布的尾部数据
# 可以用对数变换来解决这个问题

#进行对数变换
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#绘制变换后的直方图和正太概率图
# sns.distplot(df_train['SalePrice'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'], plot=plt)
# plt.show()

#2、GrLivArea(地面上总面积)
#绘制直方图和正太概率曲线图
# sns.distplot(df_train['GrLivArea'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['GrLivArea'], plot=plt)
# plt.show()
#从图中可以看出跟SalePrice有相同的特征

#进行对数变换
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#绘制变换后的直方图和正太概率图
# sns.distplot(df_train['GrLivArea'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['GrLivArea'], plot=plt)
# plt.show()

#3、TotalBsmtSF(地下室总面积)
#绘制直方图和正太概率曲线图
# sns.distplot(df_train['TotalBsmtSF'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
# plt.show()
#从图中可以看出：显示出了偏度；大量为0的观察值(没有地下室的房屋)；含0的数据无法进行对数变换
#建立一个变量，可以得到有没有地下室的影响值（二值变量），选择忽略零值，只对非零值进行对数变换，这样既可以变换数据，也不会损失有没有地下室的影响。
df_train['HasBsmt']= pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
# print(df_train.info())
# print(df_train.head(50))
#进行对数变换
df_train['TotalBsmtSF']= np.log(df_train['TotalBsmtSF'])
#绘制变换后的直方图和正态概率图
TmpTotalBsmtSF = df_train[df_train['TotalBsmtSF']>0]
TmpTotalBsmtSF = TmpTotalBsmtSF['TotalBsmtSF']
# sns.distplot(TmpTotalBsmtSF, fit=norm);
# fig = plt.figure()
# res = stats.probplot(TmpTotalBsmtSF, plot=plt)
# plt.show()

#同方差性
#最好的测量两个变量的同方差性的方法就是图像
#1、SalePrice和GrLivArea同方差性
#绘制散点图
# plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
# plt.show()

#2、SalePrice和TotalBsmtSF同方差性
#绘制散点图
# plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice'])
# plt.show()
#可以看出，SalePrice在整个TotalBsmtSF变量范围内显示出了同等级别的变化

#虚拟变量
#将类别变量转换为虚拟变量
df_train = pd.get_dummies(df_train)
# print(df_train.info())
# print(df_train.head())

# 整个方案中，使用了很多《多元数据分析》中提出的方法。
# 对变量进行了哲学分析，不仅对’SalePrice’进行了单独分析，
# 还结合了相关程度最高的变量进行分析。
# 处理了缺失数据和异常值，我们验证了一些基础统计假设，并且将类别变量转换为虚拟变量。
df_train.to_csv('./data/handled/train_outlier_dummies.csv')