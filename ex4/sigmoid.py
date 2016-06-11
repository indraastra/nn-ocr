from scipy.special import expit

sigmoid = expit

def sigmoid_gradient(x):
    y = expit(x)
    return y * (1 - y)
