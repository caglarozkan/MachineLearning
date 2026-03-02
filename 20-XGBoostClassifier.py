import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("20-digitalskysurvey.csv")

#print(df.head())
#print(df.isnull().sum())
#print(df["class"].unique()) #star - galaxy - qso

columns_to_drop=["objid","specobjid","run","rerun","camcol","field"] #redundant columns accorfing to informations about csv
df.drop(columns_to_drop,axis=1,inplace=True)

print(df.info())
print(df["class"].value_counts())

"""sns.scatterplot(x=df["redshift"],y=df["ra"],hue=df["class"]) #redshift important!!
plt.show()

sns.scatterplot(x=df["redshift"],y=df["dec"],hue=df["class"])
plt.show()

sns.scatterplot(x=df["redshift"],y=df["plate"],hue=df["class"])
plt.show()"""

df=df[df["redshift"] < 3] #cleaning outlier values

"""sns.scatterplot(x=df["redshift"],y=df["ra"],hue=df["class"]) #redshift important!!
plt.show()

sns.scatterplot(x=df["redshift"],y=df["dec"],hue=df["class"])
plt.show()

sns.scatterplot(x=df["redshift"],y=df["plate"],hue=df["class"])
plt.show()"""

#Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
df["class"]=labelencoder.fit_transform(df["class"])
print(df.head())

print(df.corr)

fig , axes = plt.subplots(nrows=1,ncols=3,figsize=(16,4))
ax=sns.histplot(df[df["class"]==0].redshift,ax=axes[0])
ax.set_title("GALAXY")
ax=sns.histplot(df[df["class"]==1].redshift,ax=axes[1])
ax.set_title("QSO")
ax=sns.histplot(df[df["class"]==2].redshift,ax=axes[2])
ax.set_title("STAR")
plt.show()


#dependent - independent feature
X=df.drop("class",axis=1)
y=df["class"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from xgboost import XGBClassifier
xgb_classifier=XGBClassifier(n_estimators=100)
xgb_classifier.fit(X_train,y_train)
y_pred=xgb_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))

#Hyperparameter tuning

params={
    "n_estimators" : [1,10,100],
    "learning_rate" : [0.1,1,10],
    "subsample" : [0.1,1,10],
    "max_depth" : [5,12,32,60]
}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=xgb_classifier,param_grid=params)
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)
print("Accuracy:",grid.score(X_test,y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))

print("Best Parameters:",grid.best_params_)
