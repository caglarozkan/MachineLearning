import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from narwhals.selectors import categorical

warnings.filterwarnings("ignore")

df=pd.read_csv("17-cardekho.csv")

#print(df.head())
#print(df.columns)

df=df.drop("Unnamed: 0", axis=1)

#print(df.describe())
#print(df.info())

print(df["seats"].value_counts()) # 2 tane 0 koltuklu arac var
print(df.loc[(df["seats"]==0)]) #3217 ve 12619 indexlerinde seats=0
df["seats"]=df["seats"].replace(0,5)
print(df["seats"].value_counts())

categorical_features=["car_name","brand","model","seller_type","fuel_type","transmission_type"]
#for col in categorical_features:
#    print(f"{col} values: {df[col].value_counts()}")

print(df[df.duplicated()])
df=df.drop_duplicates(keep="first",ignore_index=True)
print(df[df.duplicated()])

pd.set_option('display.float_forma', "{:.2f}".format)
print(df.describe())

sns.scatterplot(df,x="vehicle_age",y="selling_price",hue="fuel_type")
plt.title("Age-Price(before)")
plt.show()

sns.scatterplot(df,x="km_driven",y="selling_price",hue="fuel_type")
plt.title("KM - Price(before")
plt.show()

#outlier veriler var onları temizleyip daha uygun bir sonuc yapmamız lazım
df=df[(df["selling_price"]<1000000)]
print(df["selling_price"].max())

sns.scatterplot(df,x="vehicle_age",y="selling_price",hue="fuel_type")
plt.title("Age-Price(after)")
plt.show()

df=df[(df["km_driven"]<600000 )]
sns.scatterplot(df,x="km_driven",y="selling_price",hue="fuel_type")
plt.title("KM - Price(after)")
plt.show()


X=df.drop("selling_price",axis=1)
y=df["selling_price"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

unique_values_for_categorical=df[categorical_features].nunique()
print(unique_values_for_categorical)


#seller_type , fuel_type , transmission_type -> onehatencoding
#car_name , brand , model -> frequency encoding (ordinal encoding->)
one_hot_cols=["seller_type","fuel_type","transmission_type"]
freq_cols=["car_name","brand","model"]

for col in freq_cols:
    freq=X_train[col].value_counts()/len(X_train)

    X_train[col + "freq"]=X_train[col].map(freq) #fit_transform
    X_test[col+"freq"]=X_test[col].map(freq)  #transform mantıgı

    mean_freq = freq.mean()
    X_test[col+"freq"]= X_test[col+"freq"].fillna(mean_freq)
#print(X_train.head())
#print(X_test.head())

X_train=X_train.drop(["car_name","brand","model"],axis=1)
X_test=X_test.drop(["car_name","brand","model"],axis=1)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer=ColumnTransformer(
    transformers=[
        ("onehot",OneHotEncoder(drop="first",handle_unknown="ignore"),one_hot_cols)
    ],remainder="passthrough"
)
transformer.fit(X_train)
X_train=transformer.transform(X_train)
X_test=transformer.transform(X_test)

encoded_columns=transformer.get_feature_names_out()
print(encoded_columns)

X_train=pd.DataFrame(X_train,columns=encoded_columns)
X_test=pd.DataFrame(X_test,columns=encoded_columns) #dataframe yapmak zorunda degiliz yapmazsak numpy array olur

print(X_train.info())

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

model_ada=AdaBoostRegressor()
model_ada.fit(X_train,y_train)

y_pred=model_ada.predict(X_test)

print("R2 Score:",r2_score(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("MAE:",mean_absolute_error(y_test,y_pred))

params={
    "n_estimators": [10,50,100,200],
    "learning_rate": [0.1,1,4,5,10],
    "loss": ["linear","square","exponential"]
}

random_search=RandomizedSearchCV(estimator=AdaBoostRegressor(),param_distributions=params,cv=5,n_jobs=-1)
random_search.fit(X_train,y_train)
print("Best params for random sear:",random_search.best_params_)
y_pred=random_search.predict(X_test)

print("R2 Score:",r2_score(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("MAE:",mean_absolute_error(y_test,y_pred))