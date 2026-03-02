import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("19-heart.csv")

print(df.head())
print(df.describe())
print(df.columns)
print(df.info())

X=df.drop("target",axis=1)
y=df["target"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

gbc=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1)
gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print("Accuracy Score: ",accuracy_score(y_test,y_pred))
print("Confusion Matrix: ",confusion_matrix(y_test,y_pred))
print("Classification Report: ",classification_report(y_test,y_pred))

params={
    "n_estimators": [1,10,100,150,200],
    "learning_rate": [0.01,0.1,0.3,0.4,0.5],
    "max_depth": [1,2,3,4]
}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=params,cv=5)
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)

print("Best params: ",grid.best_params_)

print("Accuracy Score: ",grid.score(X_test,y_test))
print("Confusion Matrix: ",confusion_matrix(y_test,y_pred))
print("Classification Report: ",classification_report(y_test,y_pred))