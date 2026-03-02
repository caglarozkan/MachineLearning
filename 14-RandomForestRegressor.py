import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.svm import SVR

warnings.filterwarnings("ignore")

df=pd.read_csv("14-gym_crowdedness.csv")
#print(df.head())
#print(df.columns)
#print(df.isnull().sum())
#print(df.info())

'''pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'])
print(df["date"]'''

df["date"]=pd.to_datetime(df['date'],utc=True)
print("type of date column:",df["date"].dtype) # artık date türünde oluyor

df["year"] = df["date"].dt.year  # month,day,hour orijinal dataset'te oldugu için sadece yearı alıyoruz
#df["month"] = df["date"].dt.month # bu şekilde erişebiliyoruz diger degerlere de
#df["day"] = df["date"].dt.day
#df["hour"] = df["date"].dt.hour
print("unique value of year:",df["year"].unique())
df=df.drop("date",axis=1)

sns.lineplot(df,x="hour",y="number_people",ci=None)
plt.title("Saatlik ortalama insan")
plt.show()

sns.barplot(df,x="day_of_week",y="number_people",ci=None)
plt.title("Günlük ortalama insan")
plt.show()

df=df.drop("timestamp",axis=1)

X=df.drop("number_people",axis=1)
y=df["number_people"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#REGRESSOR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso,Ridge

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score


def calc_model_metrics(true,predicted):
    print("MSE Score:",mean_squared_error(true,predicted))
    print("R2 Score:",r2_score(true,predicted))
    print("RMSE:",np.sqrt(mean_squared_error(true,predicted)))
    print("MAE Score:",mean_absolute_error(true,predicted))

models={
    "Linear Regression":LinearRegression(),
    "Lasso Regression":Lasso(),
    "Ridge Regression":Ridge(),
    "Decision Tree Regressor":DecisionTreeRegressor(),
    "Random Forest Regressor":RandomForestRegressor(),
    "KNN Regressor":KNeighborsRegressor()
}

for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)

    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)
    print("\nModel:",list(models.values())[i]) #!!!
    print("TRAIN")
    calc_model_metrics(y_train,y_train_pred)
    print("-------------------")
    print("TEST")
    calc_model_metrics(y_test,y_test_pred)
    print("-------------------")


#Hyperparameter tunings
knn_params={
    "n_neighbors":[1,2,3,4,5],
    "weights":["uniform","distance"],
}

rfr_params={
    "n_estimators": [3,5,10,100],
    "max_features": ["sqrt","log2",5,50],
    "max_depth":[5,10,15,30]
}

from sklearn.model_selection import RandomizedSearchCV
randomcv_models=[
    ("KNN",KNeighborsRegressor(),knn_params),
    ("Random Forest",RandomForestRegressor(),rfr_params)
]
for name,model,params in randomcv_models:
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=params,n_iter=10,random_state=42,n_jobs=-1)
    randomcv.fit(X_train,y_train)
    print("best params for model:",name,randomcv.best_params_)

models={
    "Random Forest Regressor":RandomForestRegressor(
        n_estimators=10,
        max_features=5,
        max_depth=30,
    ),
    "KNN Regressor":KNeighborsRegressor(
        weights="distance",
        n_neighbors=5,
    )
}

for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)

    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)
    print("\nModel:",list(models.values())[i]) #!!!
    print("TRAIN")
    calc_model_metrics(y_train,y_train_pred)
    print("-------------------")
    print("TEST")
    calc_model_metrics(y_test,y_test_pred)
    print("-------------------")

