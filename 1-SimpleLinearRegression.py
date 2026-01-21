import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("1-studyhours.csv")
#print(df)
#print(df.info())
#print(df.describe())


plt.scatter(df["Study Hours"], df["Exam Score"])
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()

#Independent and dependent features
#X = df["Study Hours"]
#print(type(X)) #Series -> 1 input, input need to be dataframe(girdiler dataframe olmalı)
X = df[["Study Hours"]]
print(type(X))
y = df["Exam Score"] #(output series olmalı)
print(type(y)) #Series

# TEST - TRAIN split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 15)
#X'i 2ye bölüyor %80 train %20 test; y-> %80 train %20test
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

#Standardization  (!!standardization vs normalization)
#balanced feature values
#efficient gradient descent
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #Scaler kullanma sebebi columnlar arasında büyük farklı deger farklılıklarını gidermek icin
X_train = scaler.fit_transform(X_train) #trainin test ile ilgili hicbir sey bilmesini istemiyoruz, data leakage önlemek
X_test = scaler.transform(X_test)
print(X_train)
#fit-> mu= mean of the feature, sigma=standard deviation of feature
#transform-> bunları kullanarak standart hale(kucultulmus hale) cevirme

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
print("Coefficients: \n", regression.coef_)
print("Intercept: \n", regression.intercept_)  #y= 76.9076923076923 + 16.17860223 x

plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Linear Regression of Study-Note")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.savefig("1-studyhours.png")
plt.show()

#print(regression.predict(scaler.transform([[15]]))) # predict for 15 hours

#MSE - MAE - RMSE
from sklearn.metrics import mean_squared_error,mean_absolute_error ,r2_score
y_pred_test=regression.predict(X_test)

mse=mean_squared_error(y_test,y_pred_test)
mae=mean_absolute_error(y_test,y_pred_test)
rmse=np.sqrt(mse)
print("mse:",mse)
print("mae:",mae)
print("rmse:",rmse)