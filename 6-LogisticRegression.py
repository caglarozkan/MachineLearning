import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df=pd.read_csv('6-bank_customers.csv')

#print(df.head())
#print(df.describe())
#print(df.info())
#print(df.columns)

X = df.drop("subscribed",axis=1)
y = df["subscribed"] #

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.27,random_state=32)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
score = accuracy_score(y_pred,y_test)
print("\nScore:",score)
print(classification_report(y_pred,y_test))
print("Confusion Matrix")
print(confusion_matrix(y_pred,y_test))
#[ TP  ,  FP
#  FN  ,  TN ]  Confusion matrix

#hyperparameter tuning
model=LogisticRegression()
#algoritmanın accuracy'ni degiştirebilecek parametreler
penalty=["l1","l2","elasticnet"]
c_values=[100,10,1,0.1,0.01]
solver=["newton-cg","liblinear","lbfgs","sag","saga","newton-cholesky"]

params=dict(penalty=penalty,C=c_values,solver=solver)
print(params)

#grid search cv
from sklearn.model_selection import GridSearchCV,StratifiedKFold
cv=StratifiedKFold(n_splits=5)
grid=GridSearchCV(estimator=model,param_grid=params,scoring="accuracy",cv=cv,n_jobs=-1)
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

from sklearn.model_selection import RandomizedSearchCV
randomcv= RandomizedSearchCV(estimator=model,param_distributions=params,cv=cv,n_iter=10,scoring="accuracy")
randomcv.fit(X_train,y_train)

print("\nrandomcv best params:",grid.best_params_) # best parameters
print("randomcv best score:",grid.best_score_)

y_pred=grid.predict(X_test)
#print(y_pred)
score = accuracy_score(y_pred,y_test)
print("\nScore:",score)
print(classification_report(y_pred,y_test))
print("Confusion Matrix")
print(confusion_matrix(y_pred,y_test))