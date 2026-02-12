import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#email_type -> 0 = personal , 1=work
df=pd.read_csv("9-email_classification_svm.csv")
#print(df.head())
#print(df.describe())
sns.scatterplot(x=df["subject_formality_score"],y=df["sender_relationship_score"],hue=df["email_type"])
plt.show()
print(df["email_type"].value_counts())

X=df.drop("email_type",axis=1)
y=df["email_type"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30)

from sklearn.svm import SVC
svc = SVC(kernel='linear',random_state=0)
svc.fit(X_train,y_train)
print("coefs:",svc.coef_)

y_pred = svc.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print("Accuracy:",accuracy_score(y_pred,y_test))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_pred,y_test))
#veriler linear olarak çok iyi bir şekilde ayrıldıgı için burada kullandıgımız linear svc sayesinde accuracy cok yüksek cıktı

rbf=SVC(kernel="rbf",random_state=0) #kernelin default degeri rbf'dir
rbf.fit(X_train,y_train)
y_pred = rbf.predict(X_test)

print("Accuracy:",accuracy_score(y_pred,y_test))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_pred,y_test))
