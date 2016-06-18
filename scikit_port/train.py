import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from train import load_data

X, y = load_data('data/ex4data1_conv.mat', 'numpy')

classifier = LogisticRegression(C=1)
classifier.fit(X, y)

z = classifier.predict(X)
print(sum(y == z) / y.size)
