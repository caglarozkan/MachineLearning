import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv('8-fraud_detection.csv')
print(df["is_fraud"].unique())
print(df["is_fraud"].value_counts())
#print(df.head())
#print(df.tail())

X=df.drop("is_fraud",axis=1)
y=df["is_fraud"]

sns.scatterplot(x=X["transaction_amount"],y=X["transaction_risk_score"],hue=y)
plt.savefig("8-Fraud or Not")
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=20)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

penalty=["l1","l2","elasticnet"]
c_values=[100,10,1,0.1,0.01]
solver=["newton-cg","liblinear","sag","saga","lbfgs","newton-cholesky"]
class_weight=[{0:w,1:y} for w in [1,10,50,100] for y in [1,10,50,100]]

params=dict(penalty=penalty,solver=solver,class_weight=class_weight,C=c_values)

from sklearn.model_selection import GridSearchCV,StratifiedKFold
cv=StratifiedKFold()
grid = GridSearchCV(estimator=model,param_grid=params,scoring="accuracy",cv=cv)
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("Accuracy:",accuracy_score(y_pred,y_test))
print("Confusion Matrix",confusion_matrix(y_pred,y_test))
print("Classification Report",classification_report(y_pred,y_test))

print("Grid best param",grid.best_params_)
print("Grid best score",grid.best_score_)

model_prob=grid.predict_proba(X_test)
model_prob=model_prob[:,1] #probabilities for positive fraud class
from sklearn.metrics import roc_curve,roc_auc_score
model_auc=roc_auc_score(y_test,model_prob)
print("model auc:",model_auc)

#model false positive rate ,model true positive rate
model_fpr, model_tpr, thresholds = roc_curve(y_test,model_prob)
plt.plot(model_fpr,model_tpr,label="ROC curve",marker=".")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show( )