import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

from fontTools.misc.arrayTools import scaleRect

warnings.filterwarnings("ignore")

df = pd.read_csv("10-diamonds.csv")
#print(df.head())
#print(df.columns)
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())

df=df.drop("Unnamed: 0", axis=1)
print(df["x"]==0)
print("Wrong X nums: ",df[df["x"]==0])
print("\nWrong Y nums: ",df[df["y"]==0])
print("\nWrong Z nums: ",df[df["z"]==0])

print(len(df[df["x"]==0]),len(df[df["y"]==0]),len(df[df["z"]==0]))

df=df.drop(df[df["x"]==0].index)
df=df.drop(df[df["y"]==0].index)
df=df.drop(df[df["z"]==0].index)

print("Wrong X nums: ",df[df["x"]==0])
print("\nWrong Y nums: ",df[df["y"]==0])
print("\nWrong Z nums: ",df[df["z"]==0])

print("wrong nums(x y z):",len(df[df["x"]==0]),len(df[df["y"]==0]),len(df[df["z"]==0]))
#sns.pairplot(df)
#plt.savefig("10-diamonds.png")
#plt.show()

sns.scatterplot(x=df["y"],y=df["price"])
plt.title("y - price")
plt.show()
sns.scatterplot(x=df["z"],y=df["price"])
plt.title("z - price")
plt.show()
sns.scatterplot(x=df["table"],y=df["price"])
plt.title("table - price")
plt.show()
sns.scatterplot(x=df["depth"],y=df["price"])
plt.title("depth - price")
plt.show()

#outlier elimination
df =df[(df["depth"]<75) & (df["depth"]>45)]
df =df[(df["table"]<75) & (df["table"]>40)]
df =df[(df["z"]<30) & (df["z"]>2)]
df =df[df["y"]<20]

sns.scatterplot(x=df["y"],y=df["price"])
plt.title("y - price(new)")
plt.show()
sns.scatterplot(x=df["z"],y=df["price"])
plt.title("z - price(new)")
plt.show()
sns.scatterplot(x=df["table"],y=df["price"])
plt.title("table - price(new)")
plt.show()
sns.scatterplot(x=df["depth"],y=df["price"])
plt.title("depth - price(new)")
plt.show()


print("Cut Values:",df["cut"].value_counts())
print("Color Values:",df["color"].value_counts())
print("Clarity:",df["clarity"].value_counts())

X=df.drop("price",axis=1)
y=df["price"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
from sklearn.preprocessing import LabelEncoder
label_encoder= LabelEncoder()
for col in ["cut","color","clarity"]:
    X_train[col]=label_encoder.fit_transform(X_train[col])
    X_test[col]=label_encoder.transform(X_test[col]) #test'te sadece transform yapmak data leakage önlüyor

#print("Cut Values:",X_train["cut"].value_counts())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
liner_reg=LinearRegression()
liner_reg.fit(X_train_scaled,y_train)
y_pred=liner_reg.predict(X_test_scaled)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("MAE: ",mae)
print("MSE: ",mse)
print("R2: ",r2)
plt.scatter(y_test,y_pred)
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.show()


#SVR
from sklearn.svm import SVR
svr=SVR()
svr.fit(X_train_scaled,y_train)
y_pred=svr.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("MAE: ",mae)
print("MSE: ",mse)
print("R2: ",r2)
plt.scatter(y_test,y_pred)
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.show()

from sklearn.model_selection import GridSearchCV
params={
    "C":[0.1,1,10],
    "kernel": ["linear","rbf"],
    "gamma":[0.01,0.1],
}
grid = GridSearchCV(estimator=SVR(),param_grid=params,n_jobs=-1,verbose=3)
grid.fit(X_train_scaled,y_train)
print("grid best params:",grid.best_params_)
y_pred=grid.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("MAE: ",mae)
print("MSE: ",mse)
print("R2: ",r2)
plt.scatter(y_test,y_pred)
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.show()