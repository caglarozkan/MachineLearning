import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df =pd.read_csv("9-loan_risk_svm.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

sns.scatterplot(x=df["credit_score_fluctuation"],y=df["recent_transaction_volume"],hue=df["loan_risk"])
plt.show()

X=df.drop("loan_risk",axis=1)
y=df["loan_risk"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

from sklearn.svm import SVC
#linear
print("---- LINEAR ----")
linear_svc=SVC(kernel="linear",random_state=0)
linear_svc.fit(X_train,y_train)
y_pred=linear_svc.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
print("Accuracy Score:",accuracy_score(y_pred,y_test))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_pred,y_test))

#RBF
print("---- RBF ----")
rbf_svc=SVC(kernel="rbf",random_state=0)
rbf_svc.fit(X_train,y_train)
y_pred=rbf_svc.predict(X_test)
print("Accuracy Score:",accuracy_score(y_pred,y_test))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_pred,y_test))

#poly
print("---- POLY ----")
poly_svc=SVC(kernel="poly",random_state=0)
poly_svc.fit(X_train,y_train)
y_pred=poly_svc.predict(X_test)
print("Accuracy Score:",accuracy_score(y_pred,y_test))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_pred,y_test))

#sigmoid
print("---- SIGMOID ---- ")
sigmoid_svc=SVC(kernel="sigmoid",random_state=0)
sigmoid_svc.fit(X_train,y_train)
y_pred=sigmoid_svc.predict(X_test)
print("Accuracy Score:",accuracy_score(y_pred,y_test))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_pred,y_test))

param_grids={
    "C":[0.01,0.1,1,10,100],
    "kernel":["linear","rbf"],
    "gamma":[0.001,0.01,0.1,1,10,100], # can be auto,scale
}

from sklearn.model_selection import GridSearchCV
print("---- GRID SEARCH ----")
grid = GridSearchCV(estimator=SVC(),param_grid=param_grids,cv = 5)
grid.fit(X_train,y_train)
print("Best Parameters:",grid.best_params_) #fit etmeden kullanamıyoruz fit ettikten sonra best parameterslara bakmak lazım
y_pred=grid.predict(X_test)#grid'den gelen best parameterlarla tekrar bi predict yapıyo
print("Accuracy Score:",accuracy_score(y_pred,y_test))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_pred,y_test))

