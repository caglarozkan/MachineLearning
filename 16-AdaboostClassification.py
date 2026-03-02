import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("16-diabetes.csv")

#print(df.head())
#print(df.describe())
#print(df.info())
#print(df.columns)
#print(df.isnull().sum())

columns_to_check=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

for column in columns_to_check:
    zero_count=(df[column]==0).sum()
    zero_percentage=zero_count*100 / len(df[column])
    print(f"{column}: {zero_count} % {zero_percentage:.2f}")
#insulin has %48.70 0 value, drop mu etmemiz lazım yoksa NaN olarka degiştirmemiz mi

X=df.drop("Outcome",axis=1)
y=df["Outcome"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

columns_to_fill=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
medians={}
for column in columns_to_fill:
    median_value=X_train[X_train[column]!=0][column].median()
    medians[column]=median_value
    X_train[column]=X_train[column].replace(0,median_value)

for column in columns_to_fill:
    X_test[column]=X_test[column].replace(0,medians[column])

print(X_train.describe())

#Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ada=AdaBoostClassifier()
ada.fit(X_train,y_train)
y_pred=ada.predict(X_test)

print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))

#hyperparameter tuning
from sklearn.model_selection import GridSearchCV
params={
    "n_estimators":[5,10,50,100,200],
    "algorithm":["SAMME"],
    "learning_rate":[0.001,0.01,0.1,1,10],
}

grid=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=params,cv=5,n_jobs=-1,scoring="recall")
grid.fit(X_train,y_train)

print("Grid best Params: ",grid.best_params_)

print("---- After Best Params ----")
ada=AdaBoostClassifier(n_estimators=100,learning_rate=0.1,algorithm="SAMME")
ada.fit(X_train,y_train)
y_pred=ada.predict(X_test)

print("Accuracy Score",accuracy_score(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))


