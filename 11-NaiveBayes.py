import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("11-iris.csv")
#print(df.head())
#print(df.columns)
#print(df.isnull().sum())
print(df["Species"].value_counts())

sns.pairplot(df,hue="Species")
plt.savefig("11-iris.png")
plt.show()

df=df.drop("Id",axis=1)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["Species"] = label_encoder.fit_transform(df["Species"])
print(df.head())
print(df["Species"].value_counts())

X=df.drop("Species",axis=1)
y=df["Species"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#naive bayes kullanacaksak independent featureslarda scaling yapmaya gerek yok
#Gaussian kullanırken yaparsak daha iyi olur

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.naive_bayes import GaussianNB
gaussiannb=GaussianNB() #parameters-> 1-priors , 2-var_smoothing
gaussiannb.fit(X_train_scaled,y_train)
y_pred=gaussiannb.predict(X_test_scaled)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Confusion Matrix: ",confusion_matrix(y_test,y_pred))
print("Classification Report: ",classification_report(y_test,y_pred))
