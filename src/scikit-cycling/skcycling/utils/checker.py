""" Helper function to check data conformity
"""

import numpy as np


def _check_X(X):
    """ Private function helper to check if X is of proper size

    Parameters
    ----------
    X : array-like, shape = (data, )
    """

    # Check that X is a numpy vector
    if len(X.shape) is not 1:
        raise ValueError('The shape of X is not consistent. It should be a 1D numpy vector.')
    # Check that X is of type float
    if X.dtype is not 'np.float64':
        X = X.astype(np.float64)

    return X
