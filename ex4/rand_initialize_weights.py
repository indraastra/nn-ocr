import numpy as np

def rand_initialize_weights(L_in, L_out, epsilon=0.12):
    """
    Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections,
    including bias units.
    """
    return (np.random.rand(L_out, 1 + L_in) * 2 * epsilon) - epsilon
