#https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('13-car_evaluation.csv')
#print(df.head(),"\n")
#print(df.describe(),"\n")
#print(df.columns,"\n")
#print(df.isnull().sum(),"\n")

col_names=["buying","maintain","doors","persons","lug_boost","safety","class"]
df.columns=col_names
print(df.columns)
print(df.info())

#class-> target variable | doors , persons -> nunmerical | geri kalanlar -> categorical

#print(df["doors"].unique())
df["doors"]=df["doors"].replace("5more","5")
#print(df["doors"].unique())
df["doors"]=df["doors"].astype(int)
print("doors nums:",df["doors"].unique())

df["persons"]=df["persons"].replace("more","5")
df["persons"]=df["persons"].astype(int)
print("persons nums:",df["persons"].unique())
#print(df.info())
#doors and persons are int64 type

X=df.drop("class",axis=1)
y=df["class"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#print(df)

#Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_cols=["buying","maintain","lug_boost","safety"]
numerical_cols=["doors","persons"]

ordinal_encoder = OrdinalEncoder(categories=[
    ["low","med","high","vhigh"], #buying
    ["low","med","high","vhigh"], #maint
    ["small","med","big"], #lug_boot
    ["low","med","high"] #safety
])

preprocessor = ColumnTransformer(
    transformers=[
        ("transformation_doestn_matter",ordinal_encoder,categorical_cols),
    ], remainder="passthrough"
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print(pd.DataFrame(X_train_transformed))

from sklearn.tree import DecisionTreeClassifier
dec_tree=DecisionTreeClassifier(criterion="gini",random_state=15,max_depth=3)
dec_tree.fit(X_train_transformed,y_train)
y_pred=dec_tree.predict(X_test_transformed)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))

plt.figure(figsize=(12,8))
from sklearn import tree
tree.plot_tree(dec_tree.fit(X_train_transformed,y_train))
plt.show()


param={
    'criterion':['gini','entropy',"logloss"],
    'max_depth':range(1,6),
    "splitter":['best',"random"],
}

#hypermeter tuning
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=param,cv=5,scoring="accuracy")
grid_search.fit(X_train_transformed,y_train)
y_pred=grid_search.predict(X_test_transformed)

print("Best params:",grid_search.best_params_)
print("Best score:",grid_search.best_score_)

print("Accuracy:",grid_search.score(X_test_transformed,y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))


tree.plot_tree(grid_search.best_estimator_)
plt.savefig("DecisionTreeAfterGriding.png")
plt.show()