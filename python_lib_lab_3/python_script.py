import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_boston
city_boston = load_boston()
print(city_boston.keys())
data = city_boston["data"]
print(data.shape)
feature_names = city_boston["feature_names"]
print(feature_names)
target = city_boston["target"]
print(target[:10])
X = pd.DataFrame(data, columns=feature_names)
print(X.head())
print(X.info())
y = pd.DataFrame(target, columns=["price"])
print(y.info())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state =42)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
print(regressor.fit(X_train, y_train))
y_pred = regressor.predict(X_test)
check_test = pd.DataFrame({"y_test": y_test["price"], "y_pred": y_pred.flatten()})
print(check_test.head(10))
print(y_test, y_pred)
check_test["error"] = check_test["y_pred"] - check_test["y_test"]
print(check_test.head())
from sklearn.metrics import r2_score
r2_score_1 = r2_score(check_test["y_pred"], check_test["y_test"])
print(r2_score_1)




