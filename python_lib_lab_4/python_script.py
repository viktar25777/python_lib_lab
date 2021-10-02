import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(X_train, y_train))
print(scaler.transform(X_train))
from sklearn.manifold import TSNE
model = TSNE(n_components=2, learning_rate=250, random_state=42)
transformed = model.fit_transform(X_train)
X_axis = transformed[:, 0]
y_axis = transformed[:, 1]
plt.scatter(X_axis, y_axis)
print(plt.show())


