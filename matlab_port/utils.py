import numpy as np
import scipy.io as sio

def reshape_params(nn_params, input_layer_size, hidden_layer_size, num_labels):
    # Reshape weights from flattened param vectors.
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
            hidden_layer_size, (input_layer_size + 1))

    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
            num_labels, (hidden_layer_size + 1))

    return Theta1, Theta2


def flatten_params(Theta1, Theta2):
    return np.concatenate((Theta1.ravel(), Theta2.ravel()))


def load_data(input, source='numpy'):
    data = sio.loadmat(input);
    X = data['X']
    y = data['y'].flatten()
    # Correction for 1-based Matlab indexing.
    if source == 'matlab':
        y[y == 10] = 0
    return (X, y)


def shuffle_data(X, y):
    if y.ndim == 1:
      y = y.reshape((-1, 1))
    Z = np.c_[X, y]
    np.random.shuffle(Z)
    X, y = Z[:,:-y.shape[1]], Z[:,-y.shape[1]:].astype(np.uint8)
    if y.shape[1] == 1:
      y = y.flatten()
    return X, y


def partition_data(X, y, split):
    m = int(X.shape[0] * split)

    # Split into training and validation sets.
    X_1, X_2 = X[:m], X[m:]
    y_1, y_2 = y[:m], y[m:]

    return X_1, y_1, X_2, y_2
