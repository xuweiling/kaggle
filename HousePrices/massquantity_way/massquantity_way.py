import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
plt.style.use('ggplot')

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# 可视化探索
plt.figure(figsize=(15,8))
sns.boxplot(train.YearBuilt, train.SalePrice)
plt.show()
# 一般认为新房子比较贵，老房子比较便宜，从图上看大致也是这个趋势，由于建造年份 (YearBuilt) 这个特征存在较多的取值 (从1872年到2010年)，
# 直接one hot encoding会造成过于稀疏的数据，并且年份和SalePrice有正比关系，因此在特征工程中会将其进行数字化编码 (LabelEncoder) 。

plt.figure(figsize=(12,6))
plt.scatter(x=train.GrLivArea, y=train.SalePrice)
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)
plt.show()
#在右下角有两个具有极大GrLivArea可能是异常值,去掉他们

#去除异常点
train.drop(train[(train["GrLivArea"]>4000)&(train["SalePrice"]<300000)].index,inplace=True)

#将训练数据和测试数据按行连接起来，对其特征一起处理
full=pd.concat([train,test], ignore_index=True)
full.drop(['Id'],axis=1, inplace=True)
# print(full.shape)

#数据清洗

#缺失数据
aa = full.isnull().sum()
print(aa[aa>0].sort_values(ascending=False))


#首先根据LotArea（地块面积）和Neighborhood（Ames城市范围内的物理位置）的中位数输入LotFrontage（街道连接到房产的距离）的缺失值。 由于LotArea是一个连续的特征，我们使用qcut将它分成10个部分。
full.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count'])

full["LotAreaCut"] = pd.qcut(full.LotArea,10)#qcut是根据这些值的频率来选择箱子的均匀间隔，即每个箱子中含有的数的数量是相同的；cut将根据值本身来选择箱子均匀间隔，即每个箱子的间距都是相同的
full.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count'])

full['LotFrontage']=full.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# 由于LotArea和Neighborhood的某些组合不可用，所以我们只使用LotAreaCut。
full['LotFrontage']=full.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# 根据数据描述填充其他变量缺失值.

#平方英尺的砌体饰面区域、地下室未完成的平方英尺、地下室总面积、	车库容纳的车数、2型成品平方英尺、类型1完成平方英尺、车库面积，这些变量都是数值型变量，或者是有大小关系的数值类别
cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    full[col].fillna(0, inplace=True)          #这些值缺失，则表示没有相关的配置，因此用0填充

#泳池质量、其他类别未涵盖的其他功能、通往房产的胡同类型、栅栏质量、壁炉质量、车库质量、车库状况、车库的完成情况、车库建成年份、车库类型（位置）、指花园层墙、地下室的评估、评估地下室的高度、地下室成品区域的评级、地下室完工区域的评级、砌体贴面类型
cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    full[col].fillna("None", inplace=True)    #这些值缺失，属于类别类型，缺失可能因为主体并不存在，因此用None填充

# 用众数填充
#标识销售的一般分区分类、有齐全浴室的地下室、半浴室的地下室、公用设施、家庭功能、电气系统、厨房质量、销售类型、房屋外墙、房屋外墙（如果有多种材料）
cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
for col in cols2:
    full[col].fillna(full[col].mode()[0], inplace=True)            #这些值是每栋房子所必要的，缺失可能因为统计问题，因此用众数填充。

# 现在除了要求的值，已经没有缺失值了
print(full.isnull().sum()[full.isnull().sum()>0])

# 特征工程
#为了在这些特征上使用labelEncoder和get_dummies，把一些数值型特征转化为类别型特征
#识别销售涉及的住宅类型、有齐全浴室的地下室、半浴室的地下室、半个上等级的浴室、楼层以上的卧室、厨房、已售出月份（MM）、已售出年份（YYYY）、原始施工日期、重塑日期、低质量的平方英尺、车库建成年份
NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in NumStr:
    full[col]=full[col].astype(str)    #这些都是离散数值表示的变量，将数值类型转换为字符型。

# 现在要做一个很长的值映射列表
#现在要创建尽可能多的特征，然后再用模型选择好的特征。所以对SalePrice根据一个特征进行分组，并根据均值和中位数分类。

print(full.groupby(['MSSubClass'])[['SalePrice']].agg(['mean','median','count']))   #根据住宅类型对SalePrice进行分类，计算均值、中位数和计数。

# 基本上可以这样做（根据计算的均值、中位数的大小，按等级分类）
#           '180' : 1
#           '30' : 2   '45' : 2
#           '190' : 3, '50' : 3, '90' : 3,
#           '85' : 4, '40' : 4, '160' : 4
#           '70' : 5, '20' : 5, '75' : 5, '80' : 5, '150' : 5
#           '120': 6, '60' : 6

#不同的人在值得映射上有不同的思考，所以按照自己的经验做就好
#下面也在各个新的特征中加一个'o'，保留原始的特征，以免之后使用。

#对其他类别，根据分组的SalePrice值，进行映射
def map_values():
    full["oMSSubClass"] = full.MSSubClass.map({'180': 1,
                                               '30': 2, '45': 2,
                                               '190': 3, '50': 3, '90': 3,
                                               '85': 4, '40': 4, '160': 4,
                                               '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
                                               '120': 6, '60': 6})

    full["oMSZoning"] = full.MSZoning.map({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

    full["oNeighborhood"] = full.Neighborhood.map({'MeadowV': 1,
                                                   'IDOTRR': 2, 'BrDale': 2,
                                                   'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                   'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                   'NPkVill': 5, 'Mitchel': 5,
                                                   'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                   'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                   'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                   'StoneBr': 9,
                                                   'NoRidge': 10, 'NridgHt': 10})

    full["oCondition1"] = full.Condition1.map({'Artery': 1,
                                               'Feedr': 2, 'RRAe': 2,
                                               'Norm': 3, 'RRAn': 3,
                                               'PosN': 4, 'RRNe': 4,
                                               'PosA': 5, 'RRNn': 5})

    full["oBldgType"] = full.BldgType.map({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

    full["oHouseStyle"] = full.HouseStyle.map({'1.5Unf': 1,
                                               '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                               '1Story': 3, 'SLvl': 3,
                                               '2Story': 4, '2.5Fin': 4})

    full["oExterior1st"] = full.Exterior1st.map({'BrkComm': 1,
                                                 'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                                 'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3, 'HdBoard': 3,
                                                 'BrkFace': 4, 'Plywood': 4,
                                                 'VinylSd': 5,
                                                 'CemntBd': 6,
                                                 'Stone': 7, 'ImStucc': 7})

    full["oMasVnrType"] = full.MasVnrType.map({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

    full["oExterQual"] = full.ExterQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    full["oFoundation"] = full.Foundation.map({'Slab': 1,
                                               'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                               'Wood': 3, 'PConc': 4})

    full["oBsmtQual"] = full.BsmtQual.map({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oBsmtExposure"] = full.BsmtExposure.map({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

    full["oHeating"] = full.Heating.map({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

    full["oHeatingQC"] = full.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oKitchenQual"] = full.KitchenQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    full["oFunctional"] = full.Functional.map(
        {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

    full["oFireplaceQu"] = full.FireplaceQu.map({'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oGarageType"] = full.GarageType.map({'CarPort': 1, 'None': 1,
                                               'Detchd': 2,
                                               '2Types': 3, 'Basment': 3,
                                               'Attchd': 4, 'BuiltIn': 5})

    full["oGarageFinish"] = full.GarageFinish.map({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

    full["oPavedDrive"] = full.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})

    full["oSaleType"] = full.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                           'CWD': 2, 'Con': 3, 'New': 3})

    full["oSaleCondition"] = full.SaleCondition.map(
        {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})

    return "Done!"

map_values()
# 删除两个在训练时不必要的特征列：地块面积分组、销售价格；地块面积分组是为了计算LotFrontage（街道连接到房产的距离）的缺失值而产生，要删除，SalePrice是最终要计算的值
full.drop("LotAreaCut",axis=1,inplace=True)
full.drop(['SalePrice'],axis=1,inplace=True)

#test 删除不要紧的几个特征列（训练结果变好，但是提交结果变差，存在过拟合）
# full.drop("MasVnrArea", axis=1,inplace=True)
# full.drop("MasVnrType", axis=1,inplace=True)


# Pipeline 通过管道操作，可以指定一个程序的输出为另一个程序的输入，即将一个程序的标准输出与另一个程序的标准输入相连，这种机制就称为管道。
# 接下来创建一个管道. 一旦有了管道，就可以方便地试验不同的特征组合。
#对三个年份特征使用标签编码

class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lab = LabelEncoder()        ##对  创建年份、重塑日期、车库建成年份 使用标签编码器 标签编码器
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        return X

# 对倾斜的特征，使用log1p进行转化，之后通过get_dummies()函数进行独热编码
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self, skew=0.5):
        self.skew = skew

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_numeric = X.select_dtypes(exclude=["object"])  #排除对象类型特征
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index  #偏斜超过一定程度的特征
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X

# 创建pipeline
pipe = Pipeline([
    ('labenc', labelenc()),        #进行标签编码
    ('skew_dummies', skew_dummies(skew=1)),    #消除偏斜影响并进行独热编码
    ])

# 保存原始数据供之后使用
full2 = full.copy()
data_pipe = pipe.fit_transform(full2)
print(data_pipe.shape)
print(data_pipe.head())

#RobustScaler和标准化（StandardScaler）功能一样（公式：（原值-变量均值）/标准误，标准误=变量标准差/根号n，n为变量包含的个案数。标准化后，数据服从以0为均值，1为标准差的标准正态分布。）,
# 区别在于,它会根据中位数或者四分位数去中心化数据。 如果数据中含有异常值，那么使用均值和方差缩放数据的效果并不好。这种情况下，可以使用robust_scale和RobustScaler。
scaler = RobustScaler()

n_train=train.shape[0]    #获取训练数据的行数

X = data_pipe[:n_train]      #将处理完之后的数据中，训练数据和测试数据，分开
test_X = data_pipe[n_train:]
y= train.SalePrice

# 数据首先fit 训练数据，然后model从训练数据得到必要的变换信息，如特征方差和期望等，并保存为模型的参数，transform根据参数，
# 对训练数据做需要的变换。之后用在测试集上也不用在fit一次测试集，直接transform数据，等于训练集和测试集所做的变换是一样的。
X_scaled = scaler.fit(X).transform(X)    #对训练数据进行RobustScaler转化
y_log = np.log(train.SalePrice)          #对训练数据中的目标值进行log转化
test_X_scaled = scaler.transform(test_X)   #对测试数据进行RobustScaler转化


# 特征选择
# 上面的特征是不够的，所以我们需要更多。
# 组合不同的特征通常是一种好方法，但我们不知道应该选择哪些特征。 幸运的是有些模型可以提供特征选择，这里使用Lasso，但也可以自由选择Ridge，RandomForest或GradientBoostingTree。
lasso=Lasso(alpha=0.001)
lasso.fit(X_scaled,y_log)

# Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=1000,
#    normalize=False, positive=False, precompute=False, random_state=None,
#    selection='cyclic', tol=0.0001, warm_start=False)
FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=data_pipe.columns)
print(FI_lasso.sort_values("Feature Importance",ascending=False))

FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(20,32))    #根据特征重要性画图
plt.xticks(rotation=90)
plt.tick_params(labelsize=6)
plt.show()

# 基于特征的重要性，以及其他的常识，决定通过管道增加一些其他的特征.
class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self, additional=1):
        self.additional = additional

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.additional == 1:        #房子总面积 = 地下室总面积 + 一楼平方英尺 + 二楼平方英尺
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]   #全部的面积=地下室总面积 + 一楼平方英尺 + 二楼平方英尺 + 车库面积

        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]          #房子总面积*整体质量
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]            #地面以上生活区面积*整体质量
            X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]              #标识销售的一般分区分类*全部面积
            X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]            #标识销售的一般分区分类 + 整体质量
            X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]                 #标识销售的一般分区分类 + 创建年份
            X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]      #Ames城市范围内的物理位置*房子总面积
            X["+_oNeighborhood_OverallQual"] = X["oNeighborhood"] + X["OverallQual"]    #Ames城市范围内的物理位置 + 整体质量
            X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]        #Ames城市范围内的物理位置 + 创建年份
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]          #类型1完成平方英尺*整体质量

            X["-_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]         #家庭功能*房子总面积
            X["-_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]       #家庭功能 + 整体质量
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]                #地块面积*整体质量
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]                  #房子总面积 + 地块面积
            X["-_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]         #接近各种条件*房子面积
            X["-_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]       #接近各种条件 + 整体质量

            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]              #地下室面积=类型1完成面积 + 类型2完成面积 + 地下室未完成的面积
            X["Rooms"] = X["FullBath"] + X["TotRmsAbvGrd"]                                #房间数=全浴室数目 + 不包括浴室的房间总数
            X["PorchArea"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]    #门廊面积= 开放式门廊面积 + 封闭式门廊面积 + 三季门廊面积 + 屏幕门廊面积
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"] + X[
                "EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]     #总体面积=地下室总面积 + 一楼平方英尺 + 二楼平方英尺 + 车库面积 + 开放式门廊面积 + 屏幕门廊面积 + 三季门廊面积 + 屏幕门廊面积

            return X

# 通过管道，可以快速实验不同特征组合
pipe = Pipeline([
    ('labenc', labelenc()),
    ('add_feature', add_feature(additional=2)),
    ('skew_dummies', skew_dummies(skew=1)),
    ])

# PCA
# 原始特征，以及构建的特征，部分存在高度相关性，使模型的效果降低， PCA可以去除共线性。
#所以PCA的参数大致与特征个数相同，因为这儿的目的不是降维
full_pipe = pipe.fit_transform(full)
print(full_pipe.shape)     #输出全部数据规模

n_train=train.shape[0]
X = full_pipe[:n_train]
test_X = full_pipe[n_train:]
y= train.SalePrice

X_scaled = scaler.fit(X).transform(X)      #RobustScaler
y_log = np.log(train.SalePrice)
test_X_scaled = scaler.transform(test_X)

pca = PCA(n_components=410)
X_scaled=pca.fit_transform(X_scaled)
test_X_scaled = pca.transform(test_X_scaled)
print(X_scaled.shape, test_X_scaled.shape)     #输出训练数据、测试数据规模

# 建模 & 评估
# 交叉检验策略
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse

# 选择了13种模型，使用5折交叉验证来评估模型，模型包括：
# LinearRegression    线性回归
# Ridge               岭回归
# Lasso               Lasso
# Random Forrest                   随机森林
# Gradient Boosting Tree           渐变提升树
# Support Vector Regression                支持向量回归
# Linear Support Vector Regression         线性支持向量回归
# ElasticNet                        弹性网络
# Stochastic Gradient Descent       随机梯度下降
# BayesianRidge               贝叶斯岭
# KernelRidge                 内核岭
# ExtraTreesRegressor         极端随机森林回归
# XgBoost
models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),
          ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(n_iter=1000,eta0=1e-3),BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
          ExtraTreesRegressor(),XGBRegressor()]
names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]
for name, model in zip(names, models):
    score = rmse_cv(model, X_scaled, y_log)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))

#接下来做一些超参数调整。 首先定义一个gridsearch方法。
class grid():
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="neg_mean_squared_error")   #负均方误差  GridSearchCV 用于系统地遍历多种参数组合,通过交叉验证确定最佳效果参数。
        grid_search.fit(X, y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])

# Lasso
grid(Lasso()).grid_get(X_scaled,y_log,{'alpha': [0.0004,0.0005,0.0007,0.0009],'max_iter':[10000]})

# Ridge
grid(Ridge()).grid_get(X_scaled,y_log,{'alpha':[35,40,45,50,55,60,65,70,80,90]})

# SVR
grid(SVR()).grid_get(X_scaled,y_log,{'C':[11,13,15],'kernel':["rbf"],"gamma":[0.0003,0.0004],"epsilon":[0.008,0.009]})

# Kernel Ridge
param_grid={'alpha':[0.2,0.3,0.4], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.8,1]}
grid(KernelRidge()).grid_get(X_scaled,y_log,param_grid)

# ElasticNet
grid(ElasticNet()).grid_get(X_scaled,y_log,{'alpha':[0.0008,0.004,0.005],'l1_ratio':[0.08,0.1,0.3],'max_iter':[10000]})

# Ensemble Methods 集成方法
# Weight Average 加权平均数
# 基于根据权重计算的平均值的模型

class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self, mod, weight):
        self.mod = mod
        self.weight = weight

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model, data] * weight for model, weight in zip(range(pred.shape[0]), self.weight)]
            w.append(np.sum(single))
        return w


lasso = Lasso(alpha=0.0005, max_iter=10000)
ridge = Ridge(alpha=60)
svr = SVR(gamma=0.0004, kernel='rbf', C=13, epsilon=0.009)
ker = KernelRidge(alpha=0.2, kernel='polynomial', degree=3, coef0=0.8)
ela = ElasticNet(alpha=0.005, l1_ratio=0.08, max_iter=10000)
bay = BayesianRidge()
#基于每个模型的score，分配对应的权重。
w1 = 0.02
w2 = 0.2
w3 = 0.25
w4 = 0.3
w5 = 0.03
w6 = 0.2
weight_avg = AverageWeight(mod=[lasso, ridge, svr, ker, ela, bay], weight=[w1, w2, w3, w4, w5, w6])
score = rmse_cv(weight_avg, X_scaled, y_log)
print(score.mean())

#如果只用两个最好模型的平均，可以获得更好的交叉验证分数

weight_avg = AverageWeight(mod = [svr,ker],weight=[0.5,0.5])
score = rmse_cv(weight_avg,X_scaled,y_log)
print(score.mean())

# Stacking
#除了正常的stacking，还增加了get_oof方法，因为之后会将由stacking产生的特征和原始特征结合起来。
class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, mod, meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def fit(self, X, y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))

        for i, model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X, y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index, i] = renew_model.predict(X[val_index])       #预测结果

        self.meta_model.fit(oof_train, y)
        return self

    def predict(self, X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                      for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)

    def get_oof(self, X, y, test_X):
        oof = np.zeros((X.shape[0], len(self.mod)))
        test_single = np.zeros((test_X.shape[0], 5))
        test_mean = np.zeros((test_X.shape[0], len(self.mod)))
        for i, model in enumerate(self.mod):
            for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index], y[train_index])
                oof[val_index, i] = clone_model.predict(X[val_index])
                test_single[:, j] = clone_model.predict(test_X)
            test_mean[:, i] = test_single.mean(axis=1)
        return oof, test_mean


#必须先imputer处理缺失值,否则stacking不能工作。
a = Imputer().fit_transform(X_scaled)
b = Imputer().fit_transform(y_log.values.reshape(-1,1)).ravel()
stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)#用Lasso，Ridge，SVR，Kernel Ridge，ElasticNet，BayesianRidge作为第一层模型，Kernel Ridge作为第二层模型
score = rmse_cv(stack_model,a,b)
print(score.mean())


#接下来提取从stacking产生的特征，之后和原始特征组合起来。
X_train_stack, X_test_stack = stack_model.get_oof(a,b,test_X_scaled)
print(X_train_stack.shape, a.shape)

X_train_add = np.hstack((a,X_train_stack))  #(按列顺序)把数组给堆叠起来
X_test_add = np.hstack((test_X_scaled,X_test_stack))
print(X_train_add.shape, X_test_add.shape)

score = rmse_cv(stack_model,X_train_add,b)
print(score.mean())             #初始：0.10182468448，删除低相关度特征：0.101564012686（提交结果变差，有一定过拟合）
#在获得“X_train_stack”后，还可以为元模型进行参数调整，或者在与原始特征组合后进行参数调整， 但这工作量也很大
# 提交

# 这是最终使用的模型
stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
stack_model.fit(a,b)

pred = np.exp(stack_model.predict(test_X_scaled))   #exp()方法返回x的指数,e的x次方
result=pd.DataFrame({'Id':test.Id, 'SalePrice':pred})
result.to_csv("submission.csv",index=False)