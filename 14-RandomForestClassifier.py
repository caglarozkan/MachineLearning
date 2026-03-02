# income -> 50K> , 50K<
#kaggle notebook!!!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from narwhals.selectors import categorical

warnings.filterwarnings("ignore")

df=pd.read_csv("14-income_evaluation.csv")
#print(df.info())
#print(df.describe())
#print(df.columns)
#print(df.isnull().sum())

col_names=["age","workclass","finalweight","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
df.columns=col_names
print(df.columns)

#categorical and numerical
categorical=[col for col in df.columns if df[col].dtype=="O"]
numerical=[col for col in df.columns if df[col].dtype!="O"]
print("categorical values:",categorical)
print("numerical values:",numerical)

#for col in categorical:
#    print(df[col].value_counts())

over_40_hours=df[df["hours_per_week"]>40]
under_40_hours=df[df["hours_per_week"]<=40]
#print("over 40 hours",over_40_hours["income"].value_counts())

#Data Preparation
df["workclass"]=df["workclass"].replace(" ?",np.nan)
print("WORKCLASS:",df["workclass"].value_counts())

print("EDUCATION:",df["education"].unique())

print("MARITAL STATUS:",df["marital_status"].unique())

print("OCCUPATION:",df["occupation"].unique())
df["occupation"]=df["occupation"].replace(" ?",np.nan)
print("AFTER OCCUPATION:",df["occupation"].unique())

print("RELATIONSHIP:",df["relationship"].unique())

print("RACE:",df["race"].unique())

print("SEX:",df["sex"].unique())

print("NATIVE COUNTRY:",df["native_country"].unique())
df["native_country"]=df["native_country"].replace(" ?",np.nan)
print("AFTER NATIVE COUNTRY:",df["native_country"].unique())

print(df.isnull().sum())

#sns.pairplot(df,hue="income")
#plt.show()

X=df.drop("income",axis=1)
y=df["income"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for i in [X_train,X_test]:
    i["workclass"] = i["workclass"].fillna(X_train["workclass"].mode()[0])
    i["occupation"]=i["occupation"].fillna(X_train["occupation"].mode()[0])
    i["native_country"]=i["native_country"].fillna(X_train["native_country"].mode()[0])

#print(X_train.isnull().sum()) #NO NaN num
print("nunique values of categorical:",df[categorical].nunique())


#Target Encoding -> native_country cok fazla unique value var (target encoding sadece train üzerinden yapılmalıdır)
y_train_binary=y_train.apply(lambda x: 1 if x.strip()==">50K" else 0)
target_means=y_train_binary.groupby(X_train["native_country"]).mean() # o ülkeden gelip 50k dan fazla alanların oranı

X_train["native_country_encoded"] = X_train["native_country"].map(target_means)
X_train["native_country_encoded"] = X_train["native_country_encoded"].fillna(y_train_binary.mean())

X_test["native_country_encoded"] = X_test["native_country"].map(target_means)
X_test["native_country_encoded"] = X_test["native_country_encoded"].fillna(y_train_binary.mean())

X_train=X_train.drop("native_country",axis=1)
X_test=X_test.drop("native_country",axis=1)

print(X_train["native_country_encoded"])

#One Hot Encoding
one_hot_categories=[
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex"
]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

encoder=ColumnTransformer(
    transformers=[
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),one_hot_categories)],
        remainder="passthrough"
)

X_train_enc=encoder.fit_transform(X_train)
X_test_enc=encoder.transform(X_test)

columns=encoder.get_feature_names_out()

X_train=pd.DataFrame(X_train_enc,index=X_train.index,columns=columns)
X_test=pd.DataFrame(X_test_enc,index=X_test.index,columns=columns)

cols=X_train.columns

from sklearn.preprocessing import RobustScaler #outlier varsa ona daha dayanıklı bi scaling yöntemi
scaler=RobustScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train=pd.DataFrame(X_train,columns=cols)
#print(X_train)


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100,random_state=15)
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
print("\nAccuracy Score",accuracy_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))

"""feature_scores=pd.Series(rfc.feature_importances_,index=X_train.columns).sort_values(ascending=False) #columnların önemlilik sırasına göre sıralıyor burada önemsiz columnları cıkararak daha verimli bir sonuc elde edebiliriz
#print(feature_scores)

#X_train = X_train.drop(["cat__workclass_ Never-worked", "cat__occupation_ Armed-Forces","cat__education_ Preschool",
                       "cat__workclass_ Without-pay", "cat__occupation_ Priv-house-serv", "cat__marital_status_ Married-AF-spouse",
                        "cat__education_ 1st-4th", "cat__education_ 5th-6th", "cat__race_ Other", "cat__education_ 12th"
                       ], axis=1,errors="ignore")

#X_test = X_test.drop(["cat__workclass_ Never-worked", "cat__occupation_ Armed-Forces","cat__education_ Preschool",
                       "cat__workclass_ Without-pay", "cat__occupation_ Priv-house-serv", "cat__marital_status_ Married-AF-spouse",
                        "cat__education_ 1st-4th", "cat__education_ 5th-6th", "cat__race_ Other", "cat__education_ 12th"
                       ], axis=1,errors="ignore")"""

#hyperparameter tuning
rf_params={
    "n_estimators":[1,10,100,1000],
    "max_depth":[5,10,15],
    "max_features":["sqrt","log2",10,20],
    "min_samples_split":[2,7,12,15]
}
from sklearn.model_selection import RandomizedSearchCV

rfc_search=RandomizedSearchCV(RandomForestClassifier(),param_distributions=rf_params,cv=5,n_jobs=-1)
rfc_search.fit(X_train,y_train)
print("Rfc best params:",rfc_search.best_params_)
y_pred=rfc_search.predict(X_test)

print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))

