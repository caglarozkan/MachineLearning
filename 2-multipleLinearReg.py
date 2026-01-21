import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("2-multiplegradesdataset.csv")
#print(df.info())
#print(df.head())

df.isnull().sum()
df = df.dropna()
print(df.corr())

plt.scatter(df["Sleep Hours"], df["Exam Score"], color="blue") #Example plot
plt.xlabel("Sleep Hours")
plt.ylabel("Exam Score")
plt.show()

sns.regplot(x=df["Study Hours"], y=df["Exam Score"])
plt.savefig("2-StudyHoursVsExam(sns.regplot).png")
plt.show()

X = df[["Study Hours", "Sleep Hours", "Attendance Rate", "Social Media Hours"]] #DataFrame
y=df["Exam Score"] #Series
print("X shape:", X.shape)
print("y shape:", y.shape)

#train-test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=20)
#print("x-train",X_train)
#print("x-test",X_test)
#print("y-test",y_train)
#print("y-test",y_test)

plt.scatter(X_train["Study Hours"], y_train , color="green")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.show()

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regression = LinearRegression()
regression.fit(X_train,y_train)
print("Coefficients",regression.coef_)
print("Intercept",regression.intercept_)

new_student=[[10,3,87,5]]
new_student_scaled=scaler.transform(new_student)
print("Predicted exam score for new student:",regression.predict(new_student_scaled))

new_students = {
    "caglar": [[4, 8, 95, 7]],
    "yigit":  [[5, 7, 74, 8]],
    "yusuf":  [[6, 5, 56, 12]],
}

for name, raw in new_students.items():
    raw_arr = np.array(raw)
    raw_scaled = scaler.transform(raw_arr)
    pred = regression.predict(raw_scaled)[0]
    print(f"{name}: raw_pred={pred:.2f}")


y_predicted = regression.predict(X_test)

plt.scatter(y_test,y_predicted,color="red")
plt.xlabel(" True Exam score")
plt.ylabel("Predicted Exam Score")
plt.savefig("2-predicted_vs_real.png")
plt.show()

mse=mean_squared_error(y_test,y_predicted)
print("mse:",mse)