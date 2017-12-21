import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import ensemble

##################
# This script was written to solve the Housing Prices problem at https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# Last checked, it placed 1417th out of 2953rd on the Leaderboard.
#
# It uses all of the features in the dataset (except for Id, obviously)
# Each feature is either "discrete" or "continuous"
# For each feature, its values are mapped to an estimate of the Sale Price based solely on that value

# For discrete features, each category is replaced by the mean sale price of houses in that category
# E.g. For the Street feature, all "Paved" entries are mapped to the mean price of houses with "Paved" streets.

# For continuous features, a Linear Regression model is fitted to the column vs the SalePrice column
# When mapping the feature, the entries are simply plugged into the Linear Regression model
# The ContinuousMap class is simply a wrapper for this Linear Regression model

# When all features have their data mapped, a Gradient Boosting Regressor is applied
# to make the overall prediction of the house's selling price
##################

class ContinuousMap:

    #fits the Linear Regression model
    def __init__(self, df, input, out):
        self.ml = LinearRegression()

        X_train = df.drop(out, axis=1)
        Y_train = df[out]

        self.ml.fit(X_train, Y_train)

    #plugs value (number or 2D array with 1 feature) into model
    def get(self, input):
        return self.ml.predict(input)


#generates a map that maps from col1 to col2 in the df DataFrame
def gen_map(df, col1, col2, nullidx, nullval):
    return gen_map_h(df[[col1, col2]].groupby(col1).mean(), col2, nullidx, nullval)

#helper function for above
def gen_map_h(df, col2, nullidx, nullval):
    map = pd.Series()
    for row in df.iterrows():
        map.set_value(row[0], row[1][col2])

    map.set_value(nullidx, nullval)
    return map

#returns ContinuousMap object that maps from col1 to col2
def gen_cont_map(df, col1, col2):
    return ContinuousMap(df[[col1,col2]].groupby(col1, as_index=False).mean(), col1, col2)

#read in datasets
train = pd.read_csv("datasets/housing/train.csv")
test = pd.read_csv("datasets/housing/test.csv")

#lists all continuous features to be used
cont_feat = ['LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','\
1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','\
TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','\
ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']

#lists all discrete features to be used
discrete_features = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood', 'Condition1',\
'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual',\
'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical', \
'KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature', \
'SaleType','SaleCondition']

#creates DataFrames to hold processed info
newtrain = pd.DataFrame()
newtest = pd.DataFrame()

#maps discrete features
for feat in discrete_features:

    #fills empty entries in DataFrames with a NULL value
    train[feat] = train[feat].fillna('None')
    test[feat] = test[feat].fillna('None')

    #checks to see if the "test" DataFrame contains any values that the "train" DataFrame doesn't
    contains = test[feat].isin(train[feat])

    #sets any values in "test" not contained in "train" to zero (another NULL value)
    test.loc[~contains, feat] = 0

    #applies the map from this feature to the "SalePrice" feature
    newtrain[feat] = train[feat].map(gen_map(train, feat, 'SalePrice', 0, 0))
    newtest[feat] = test[feat].map(gen_map(train, feat, 'SalePrice', 0, 0))

#maps continuous feature
for feat in cont_feat:

    #fills empty values with new NULL value
    train[feat] = train[feat].fillna(0)
    test[feat] = test[feat].fillna(0)

    #generates continuous map
    map = gen_cont_map(train, feat, 'SalePrice')

    #applies map to feature's column of data
    newtrain[feat] = map.get(train[feat].values.reshape(-1, 1))
    newtest[feat] = map.get(test[feat].values.reshape(-1, 1))


#generates training and testing data for debugging
#currently, prop = 0 so we train with ALL of the train data
#to output the most accurate data
prop = 0.0
X_train, X_test, Y_train, Y_test = train_test_split(newtrain, train['SalePrice'], test_size=prop, random_state=0)


# Random Forest Regressor was second most effective with a RMSLE of 0.144640807671 (with untuned parameters)
# when trained on 80% of the training data and tested on the rest

# rand = ensemble.RandomForestRegressor(random_state=0, bootstrap=False, min_samples_leaf=1, n_estimators=500, min_samples_split=2, max_features=10, max_depth=None)
# rand = ensemble.RandomForestRegressor(random_state=0)
# rand.fit(X_train, Y_train)
# print math.sqrt(mean_squared_log_error(Y_test, rand.predict(X_test)))

# GradientBoostingRegressor was the most effective, with a RMSLE of 0.128108341611 (with untuned parameters)
# when trained on 80% of the training data and tested on the rest
gradboost = ensemble.GradientBoostingRegressor()
gradboost.fit(X_train, Y_train)
Y_pred = gradboost.predict(newtest)

#writes the predictions to "housing.csv" text file
write = pd.DataFrame()
write['Id'] = test['Id']
write['SalePrice'] = Y_pred

write.to_csv("predictions/housing.csv", index=False)