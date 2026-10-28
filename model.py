import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression,Ridge,Lasso,ElasticNet,SGDRegressor,HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
import pickle


#loading the dataset
data=pd.read_csv(r"C:\Users\Likit\Downloads\USA_Housing - USA_Housing.csv")

#preprocessing the data
x=data.drop(['Price','Address'],axis=1)
y=data['Price']

#splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#define the models to be used
models = {'LinearRegression': LinearRegression(),
          'Ridge': Ridge(),
          'Lasso': Lasso(),
          'ElasticNet': ElasticNet(),
          'SGDRegressor': SGDRegressor(),
          'PolynomialRegression': Pipeline([('poly', PolynomialFeatures(degree=4)), ('linear', LinearRegression())]),
          'HuberRegressor': HuberRegressor(),
          'RandomForestRegressor': RandomForestRegressor(),
          'SVR': SVR(),
          'MLPRegressor': MLPRegressor(),
          'KNeighborsRegressor': KNeighborsRegressor(),
          'LightGBM': lgb.LGBMRegressor(),
          'XGBoost': xgb.XGBRegressor()}

#training and evaluating the models
results = []

for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    results.append({'Model':name,'MSE':mse,'R2':r2,'MAE':mae})
    with open(f"{name}.pkl",'wb') as f:
        pickle.dump(model,f)
        
        
#creating a dataframe to display the results
results_df=pd.DataFrame(results)
results_df.to_csv(r"C:\Users\Likit\Downloads\model_results.csv",index=False)
print("Model training and evaluation completed. Results saved to model_results.csv")