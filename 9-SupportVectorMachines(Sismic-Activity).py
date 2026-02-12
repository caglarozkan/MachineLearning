import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("9-seismic_activity_svm.csv")
#print(df.head())
#print(df.isnull().sum())
#print(df["seismic_event_detected"].value_counts())

sns.scatterplot(x=df["underground_wave_energy"], y=df["vibration_axis_variation"], hue=df["seismic_event_detected"]) #2 boyutlu olarak çözülemez o yüzden 3 boyutlu cozmek lazım
plt.show()

print(df.columns)

df["underground_wave_energy"]=df["underground_wave_energy"]**2
df["vibration_axis_variation"]=df["vibration_axis_variation"]**2
df["underground_wave_energy*vibration_axis_variation"]=(df["underground_wave_energy"]*df["vibration_axis_variation"])

print(df.head())
X= df.drop("seismic_event_detected",axis=1)
y= df["seismic_event_detected"]

#trian - test - split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

import plotly.express as px
fig = px.scatter_3d(df,x=df["underground_wave_energy**2"],y=df["vibration_axis_variation**2"],z=df["undergroun_wave_energy*vibration_axis_variation"],color=df["seismic_event_detected"])
fig.show()
