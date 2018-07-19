import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#读数据
df_test = pd.read_csv('./data/test.csv', index_col=False)
print(df_test.columns)
#查看列信息
# print(df_test.columns)

#缺失数据
# 关于缺失数据需要思考的重要问题：
# 1、这一缺失数据的普遍性如何？
# 2、缺失数据是随机的还是有律可循？
# 这些问题的答案是很重要的，因为缺失数据意味着样本大小的缩减，这会阻止分析进程。
# 除此之外，以实质性的角度来说，需要保证对缺失数据的处理不会出现
# 偏离或隐藏任何难以忽视的真相。
total = df_test.isnull().sum().sort_values(ascending=False)   #df.isnull().sum()返回每列包含的缺失值的个数
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)  #df_test.isnull().count()df中所有值的个数
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


# BsmtFinSF1                 1  0.000685    补充均值
# BsmtFinSF2                 1  0.000685    补充均值
# GarageCars                 1  0.000685    补充均值
# GarageArea                 1  0.000685    补充均值
# TotalBsmtSF                1  0.000685    填充0
# BsmtUnfSF                  1  0.000685    补充均值


df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(0)
df_test['BsmtFinSF1'] = df_test['BsmtFinSF1'].fillna(0)
df_test['BsmtFinSF2'] = df_test['BsmtFinSF2'].fillna(0)
df_test['GarageCars'] = df_test['GarageCars'].fillna(0)
df_test['GarageArea'] = df_test['GarageArea'].fillna(0)
df_test['BsmtUnfSF'] = df_test['BsmtUnfSF'].fillna(0)


df_test = df_test.drop((missing_data[missing_data['Total'] > 1]).index, 1)#删除整列
df_test = df_test.drop(df_test.loc[df_test['Electrical'].isnull()].index)
# print(df_test.isnull().sum().max())  #检查是否还有缺失值
df_test.to_csv('./data/handled/test_misingvalue_handled.csv', index=False, encoding='utf-8')

#正态性

#进行对数变换
df_test['GrLivArea'] = np.log(df_test['GrLivArea'])

#TotalBsmtSF(地下室总面积)
df_test['HasBsmt']= pd.Series(len(df_test['TotalBsmtSF']), index=df_test.index)
df_test['HasBsmt'] = 0
df_test.loc[df_test['TotalBsmtSF']>0,'HasBsmt'] = 1
# print(df_test.info())
# print(df_test.head(50))
#进行对数变换
df_test['TotalBsmtSF'].loc[df_test['TotalBsmtSF'] == 0] = 0.001   #将0值替换为可log变换的另一个较小值
df_test['TotalBsmtSF']= np.log(df_test['TotalBsmtSF'])


#虚拟变量
#将类别变量转换为虚拟变量
df_test = pd.get_dummies(df_test)
# print(df_test.info())
# print(df_test.head())

df_test.to_csv('./data/handled/test_outlier_dummies.csv', index=False, encoding='utf-8')

# total = df_test.isnull().sum().sort_values(ascending=False)   #df.isnull().sum()返回每列包含的缺失值的个数
# percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)  #df_test.isnull().count()df中所有值的个数
# missing_data = pd.concat([total,percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(20))