import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df =pd.read_csv("18-concrete_data.csv")

print(df.columns)
print(df.describe())
print(df.info())

#sns.heatmap(df.corr(),annot=True)
#plt.show()

sns.scatterplot(df,x="Water",y="Cement",hue="Strength")
plt.show()

#dependent and independent features
X=df.drop("Strength",axis=1)
y=df["Strength"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

from sklearn.tree import DecisionTreeRegressor
#first weak learner
tree_reg1= DecisionTreeRegressor(max_depth=3)
tree_reg1.fit(X_train,y_train)
y2=y_train - tree_reg1.predict(X_train)

#second weak learner
tree_reg2=DecisionTreeRegressor(max_depth=3)
tree_reg2.fit(X_train,y2)
y3=y2 - tree_reg2.predict(X_train)

#three weak learner
tree_reg3=DecisionTreeRegressor(max_depth=3)
tree_reg3.fit(X_train,y3)
y4=y3 - tree_reg3.predict(X_train)
y_pred = sum(tree.predict(X_test) for tree in (tree_reg1,tree_reg2,tree_reg3))
print(y_pred)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print("R2 Score:",r2_score(y_test,y_pred))


from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor(n_estimators=5,max_depth=3,learning_rate=0.1) # bad score
gbr.fit(X_train,y_train)

y_pred=gbr.predict(X_test)
print("\nR2 Score:",r2_score(y_test,y_pred))


params={
    "n_estimators":[1,10,100,150,200],
    "learning_rate":[0.01,0.1,0.5,1],
    "max_depth":[2,3,4,5],
    "loss":["squared_error","huber","absolute_error","quantile"]
}

from sklearn.model_selection import RandomizedSearchCV
rscv=RandomizedSearchCV(estimator=GradientBoostingRegressor(),param_distributions=params,cv=5)
rscv.fit(X_train,y_train)
y_pred=rscv.predict(X_test)
print("Best params: ",rscv.best_params_)

print("Best score: ",rscv.best_score_)
print("R2 score:",r2_score(y_test,y_pred))