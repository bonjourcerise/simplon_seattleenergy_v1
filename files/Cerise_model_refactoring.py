import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

data=pd.read_csv("data.csv")

def fit_model(data):
    #define X & y
    y=data[['TotalGHGEmissions','SiteEnergyUse(kBtu)']]
    X=data.drop(['TotalGHGEmissions','SiteEnergyUse(kBtu)'],axis=1)
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=2022)
    # preprocessing
    num_columns = X.columns[X.dtypes != 'object']
    cat_columns = X.columns[X.dtypes == 'object']
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    numeric_transformer = Pipeline(steps=[('scaler',RobustScaler())])
    preprocessor = make_column_transformer((RobustScaler(),num_columns),(OneHotEncoder(handle_unknown = 'ignore'),cat_columns ))
    # model fit
    model_tree = Pipeline([
        ('transformer', preprocessor),
        ('model_tree',  DecisionTreeRegressor()),
    ])
    model_tree.fit(X_train,y_train)
    # pred
    y_pred = model_tree.predict(X_test)
    # print metrics
    print("MAE: ", (mean_absolute_error(y_test, y_pred)))
    print("MSE: ", (mean_squared_error(y_test,y_pred)))
    print("RMSE: ", (mean_squared_error(y_test,y_pred, squared=False)))

