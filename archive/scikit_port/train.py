import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from matlab_port.utils import partition_data, shuffle_data
from train import load_data

X, y = load_data('data/ex4data1_conv.mat', 'numpy')
X, y = shuffle_data(X, y)
X, y, X_test, y_test = partition_data(X, y, split=.8)

classifier = LogisticRegression(C=.1)
classifier.fit(X, y)

z = classifier.predict(X_test)
print("Test accuracy: {acc:.1%}".format(acc=(sum(y_test == z) / y_test.size)))
