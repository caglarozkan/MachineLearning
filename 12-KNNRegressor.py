import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("12-house_energy_regression.csv")
print(df.head())
print(df.describe())

sns.scatterplot(x=df["avg_indoor_temp_change"],y=df["outdoor_humidity_level"])
plt.show()

X=df.drop("daily_energy_consumption_kwh",axis=1)
y=df["daily_energy_consumption_kwh"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.neighbors import KNeighborsRegressor
#KNN = 5
knn = KNeighborsRegressor(n_neighbors=5,algorithm='auto',weights='uniform')
knn.fit(X_train_scaled,y_train)
y_pred = knn.predict(X_test_scaled)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print("r2 score:",r2_score(y_test,y_pred))
print("mean squared error:",mean_squared_error(y_test,y_pred))
print("mean absolute error:",mean_absolute_error(y_test,y_pred))

plt.scatter(y_pred,y_test,c="red")

#KNN = 10
knn = KNeighborsRegressor(n_neighbors=10,algorithm='auto',weights='uniform')
knn.fit(X_train_scaled,y_train)
y_pred = knn.predict(X_test_scaled)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print("r2 score:",r2_score(y_test,y_pred))
print("mean squared error:",mean_squared_error(y_test,y_pred))
print("mean absolute error:",mean_absolute_error(y_test,y_pred))

plt.scatter(y_pred,y_test,c="red")

