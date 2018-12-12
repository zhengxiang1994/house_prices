# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the data
df_train = pd.read_csv(r"../data/train.csv")
# print(df_train)


# check the decoration
# print(df_train.columns)
'''
There 79 features (not include Id), SalePrice is the output.
Id, MSSubClass, MSZoning, LotFrontage, LotArea, Street, Alley, LotShape, LandContour, Utilities, LotConfig,
LandSlope, Neighborhood, Condition1, Condition2, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt, 
YearRemodAdd, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, MasVnrArea, ExterQual, ExterCond, 
Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinSF1, BsmtFinType2, BsmtFinSF2, BsmtUnfSF, 
TotalBsmtSF, Heating, HeatingQC, CentralAir, Electrical, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, 
BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, KitchenQual, TotRmsAbvGrd, Functional, Fireplaces, 
FireplaceQu, GarageType, GarageYrBlt, GarageFinish, GarageCars, GarageArea, GarageQual, GarageCond, PavedDrive, 
WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, PoolQC, Fence, MiscFeature, MiscVal, 
MoSold, YrSold, SaleType, SaleCondition, SalePrice
'''

# check the summary description
# print(df_train["SalePrice"].describe())

# check the distribution of house price
# sns.distplot(df_train["SalePrice"])
# plt.show()

# check the relation between price and variables
# price and GrLivArea
# sns.set(style="darkgrid")
# sns.relplot(x="GrLivArea", y="SalePrice", data=df_train)
# plt.show()

# price and TotalBsmtSF
# sns.set(style="darkgrid")
# sns.relplot(x="TotalBsmtSF", y="SalePrice", data=df_train)
# plt.show()

# price and OverallQual
# sns.set(style="darkgrid")
# sns.boxplot(x="OverallQual", y="SalePrice", data=df_train)
# plt.show()

# check the correlations between two features
# corrmat = df_train.corr()
# sns.heatmap(corrmat, vmax=0.8, square=True)
# plt.show()

# process the missing data
# total_miss = df_train.isnull().sum()
# percentage = (total_miss / df_train.isnull().count()).sort_values(ascending=False)
# print(percentage.head(20))
'''
if the percentage larger than 0.15, the feature is pruned.
So pruned features: PoolQC, MiscFeature, Alley, Fence, FireplaceQu, LotFrontage
GarageCars contain the information of garage, so prune GarageCond, GarageType, GarageYrBlt, GarageFinish, GarageQual, so
as BsmtExposure, BsmtFinType2, BsmtFinType1, BsmtCond, BsmtQual.
Besides, we also prune MasVnrArea and MasVnrType. 
'''

# the only feature that has missing data is Electrical
# print(df_train["Electrical"].isnull().sum())    # only one sample has missing data, so delete the missing data


# prune some features and data
def pruneFeatureAndData(df):
    pruned_features = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "LotFrontage",
                       "GarageCond", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual",
                       "BsmtExposure", "BsmtFinType2", "BsmtFinType1", "BsmtCond", "BsmtQual",
                       "MasVnrArea", "MasVnrType"]
    # drop features
    df = df.drop(pruned_features, axis=1)
    # drop samples
    df = df.drop(df.loc[df["Electrical"].isnull()].index)
    return df


# check the types of features (discrete or continuous)
quantity = [attr for attr in df_train.columns if df_train.dtypes[attr] != "object"]
quality = [attr for attr in df_train.columns if df_train.dtypes[attr] == "object"]


def discrete2Onehot(df):
    if "SalePrice" in df.columns:
        salePrice = df["SalePrice"]
        df = df.drop("SalePrice", axis=1)
        df = pd.get_dummies(df)
        return pd.concat([df, salePrice], axis=1)
    else:
        return pd.get_dummies(df)


if __name__ == "__main__":
    # df_train = discrete2Onehot(pruneFeatureAndData(df_train))
    # print(df_train)
    # df_train.to_csv("../data/train1.csv", index=0)
    # print("save success!")

    df_test = pd.read_csv(r"../data/test.csv")
    df_test = discrete2Onehot(pruneFeatureAndData(df_test))
    df_test.to_csv("../data/test1.csv", index=0)
    print("save success!")





