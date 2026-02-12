import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("12-health_risk_classification.csv")
print(df.head())
print(df.isnull().sum())

sns.scatterplot(x=df["blood_pressure_variation"],y=df["activity_level_index"],hue=df["high_risk_flag"])
plt.show()

print(df["high_risk_flag"].value_counts()) #balanced dataset

X=df.drop("high_risk_flag",axis=1)
y=df["high_risk_flag"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,algorithm='auto',weights='uniform')
knn.fit(X_train_scaled,y_train)
y_pred = knn.predict(X_test_scaled)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print("Accuracy Score")
print(accuracy_score(y_test,y_pred))
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("Classification Report")
print(classification_report(y_test,y_pred))

print("best algorithm:",knn.algorithm)
