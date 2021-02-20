import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
df = pd.read_csv("data_linear_regression.csv")
df = df.select_dtypes(include=[np.number]).interpolate().dropna()#get rid of non numerical values
X = df.drop(["revenue"], axis=1)
y = df["revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=10)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train) #train model
pred = reg.predict(X_test) #get prediction
test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
test_set_r2 = r2_score(y_test, pred)


print(test_set_rmse)
print(test_set_r2)


#question 3
numeric_features = df.select_dtypes(include=[np.number])
corr = numeric_features.corr()
top_5_features = corr['revenue'].sort_values(ascending=False)[:5]
X_train, X_test, y_train, y_test = train_test_split(top_5_features, y, test_size=0.3,random_state=10)
test_rmse = (np.sqrt(mean_squared_error(top_5_features, y)))
test_r2 = r2_score(y_test, pred)
print(test_rmse)
print(test_r2)