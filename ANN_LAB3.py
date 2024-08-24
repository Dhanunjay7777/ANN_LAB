import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=25)
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

multi_layer = MLPClassifier(hidden_layer_sizes=(8, 8, 8), max_iter=1000, random_state=25)
multi_layer.fit(x_train, y_train)

predict = multi_layer.predict(x_test)
print(predict)

print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))



