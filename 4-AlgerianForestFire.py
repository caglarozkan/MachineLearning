import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("4-Algerian_forest_fires_dataset.csv")

#print(df.head())
#print(df.info())
#print(df.describe())

#DATA CLEANING
print(df.isnull().sum())
print(df[df.isnull().any(axis=1)]) #which rows are including NaN value

df.drop(122,inplace=True) #122 is completely NaN

df.loc[:123,"Region"] = 0
df.loc[123:,"Region"]=1

df = df.dropna().reset_index(drop=True)
print(df.isnull().sum())
#print(df.iloc[121]) #Region 0 (last element)
#print(df.iloc[122]) #Region 1 (first element)

df.columns=df.columns.str.strip()
print(df.columns)

df.drop(122,inplace=True)

df[["day","month","year","Temperature","RH","Ws"]]=df[["day","month","year","Temperature","RH","Ws"]].astype(int)
df[["Rain","FFMC","DMC","DC","ISI","BUI","FWI"]] =df[["Rain","FFMC","DMC","DC","ISI","BUI","FWI"]].astype(float)

print(df["Classes"].value_counts()) #

df["Classes"]=np.where(df["Classes"].str.contains("fire"),df["Classes"].str.contains("not fire"),1) # fire->1 not-fire->0
print(df["Classes"].value_counts(normalize=True))#Normalize -> %? %?

#sns.heatmap(df.corr())
#plt.show()
df.drop(["day","month","year"],axis=1,inplace=True) # not important columns -> day,month and year


#dependent & independent features
X=df.drop("FWI", axis =1)
y=df["FWI"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#redundancy,multicollinearity,overfitting %95ten fazla korelasyon varsa cıkarmak daha mantıklı hatta %85den fazla
print(type(X_train.corr()))

def correlation_for_dropping(df,threshold):
    columns_to_drop=set()
    corr = df.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j]) > threshold: #ALL CORRELATIONS
               columns_to_drop.add(corr.columns[i])
    return columns_to_drop

print(correlation_for_dropping(X_train,0.85)) #correlation %85ten fazla olanları siliyor
columns_to_drop=correlation_for_dropping(X_train,0.85)
X_train.drop(columns_to_drop,axis=1,inplace=True)
X_test.drop(columns_to_drop,axis=1,inplace=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title("X_train")
plt.subplot(1,2,2)
sns.boxplot(data=X_train_scaled)
plt.title("X_train_scaled")
plt.savefig("Normal-Scaled(boxplot).png")
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

linear=LinearRegression()
linear.fit(X_train_scaled,y_train)
y_pred=linear.predict(X_test_scaled)

mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mse:",mse)
print("mae:",mae)
print("r2 score:",r2)
plt.scatter(y_test,y_pred)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("y_test-y_pred")
plt.show()

from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(X_train_scaled,y_train)
y_pred=lasso.predict(X_test_scaled)

mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("----Lasso----")
print("mse(lasso):",mse)
print("mae(lasso):",mae)
print("r2 score(lasso):",r2)
plt.scatter(y_test,y_pred)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("y_test-y_pred - Lasso")
plt.show()
plt.show()

from sklearn.linear_model import Ridge
ridge=Ridge()
ridge.fit(X_train_scaled,y_train)
y_pred=ridge.predict(X_test_scaled)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("----Ridge----")
print("mse(Ridge):",mse)
print("mae(Ridge):",mae)
print("r2 score(Ridge):",r2)
plt.scatter(y_test,y_pred)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("y_test-y_pred - Ridge")
plt.show()

from sklearn.linear_model import ElasticNet
elastic=ElasticNet()
elastic.fit(X_train_scaled,y_train)
y_pred=elastic.predict(X_test_scaled)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("----ElasticNet----")
print("mse(Elastic):",mse)
print("mae(Elastic):",mae)
print("r2 score(Elastic):",r2)
plt.scatter(y_test,y_pred)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("y_test-y_pred - ElasticNet")
plt.show()


##LAZY PREDICT !!!!
