import numpy as np


def awgn(x, stddev):
    """Emulation of AWGN channel for a given stddev.

    Function adds Gaussian distributed noise to the input values with the given noise power.

    Args:
        x: array of real input values.
        stddev: standard deviation for the added noise.

    Result:
        Array of noisy channel symbols.
    """
    noise = np.random.normal(0, stddev, x.shape)
    return x + noise
