import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df=pd.read_csv("7-cyber_attack_data.csv")
#print(df.columns)
#print(df.head())
#print(df.describe())
#print(df.info())

X=df.drop("attack_type",axis=1)
y=df["attack_type"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=17)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix:",cm)
print("Accuracy Score:", accuracy_score(y_pred,y_test))
print("Classification Report:",classification_report(y_test,y_pred))

penalty=["l1","l2","elasticnet"]
c_values=[100,10,1,0.1,0.01]
solver=["newton-cg","liblinear","lbfgs","sag","saga","newton-cholesky"]

params=dict(penalty=penalty,C=c_values,solver=solver)
print(params)

#grid search cv
from sklearn.model_selection import GridSearchCV,StratifiedKFold
cv=StratifiedKFold(n_splits=5)
grid=GridSearchCV(estimator=logreg,param_grid=params,scoring="accuracy",cv=cv,n_jobs=-1)
grid.fit(X_train,y_train)
print("grid best params:",grid.best_params_) # best parameters
print("grid best score:",grid.best_score_)

y_pred=grid.predict(X_test)
#print(y_pred)
score = accuracy_score(y_pred,y_test)
print("\nScore:",score)
print(classification_report(y_pred,y_test))
print("Confusion Matrix")
print(confusion_matrix(y_pred,y_test))

#one vs rest
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
onevsone_model=OneVsOneClassifier(LogisticRegression())
onevsrest_model=OneVsRestClassifier(LogisticRegression)

onevsone_model.fit(X_train,y_train)
y_pred=onevsone_model.predict(X_test)

score = accuracy_score(y_pred,y_test)
print("One vs One")
print("\nScore:",score)
print(classification_report(y_pred,y_test))
print("Confusion Matrix")
print(confusion_matrix(y_pred,y_test))

onevsrest_model.fit(X_train,y_train)
y_pred=onevsrest_model.predict(X_test)
score = accuracy_score(y_pred,y_test)
print("One vs Rest")
print("\nScore:",score)
print(classification_report(y_pred,y_test))
print("Confusion Matrix")
print(confusion_matrix(y_pred,y_test))