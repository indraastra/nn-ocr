import numpy as np

from .sigmoid import sigmoid

def predict(Theta1, Theta2, X, raw=False):
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    p = np.zeros((m, 1));

    bias = np.ones((m, 1))
    h1 = sigmoid(np.c_[bias, X].dot(Theta1.T));
    h2 = sigmoid(np.c_[bias, h1].dot(Theta2.T));

    # Return the index of the highest-probability class, which conveniently
    # is also the class label.
    if raw:
        return h2
    else:
        return h2.argmax(axis=1);
