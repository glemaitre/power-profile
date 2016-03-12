""" Metrics to asses the performance of a cycling ride

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

from ..utils.checker import _check_X
from ..restoration.denoise import _moving_average
import numpy as np


def normalized_power_score(X):
    """ Compute the normalized power for a given ride

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    Returns
    -------
    score : float
        Return the normalized power.
    """

    # Check the conformity of X
    X = _check_X(X)

    # Compute the moving average
    x_avg = _moving_average(X, n=30)

    # Removing value < 35% PMA
    arr = np.array([[-1, 5], [1, 1], [3, 11], [-4, 20], [2, 9]])
    x_avg[~(x_avg[:] < 0.35 * pma)]

    # Compute the mean of the denoised ride elevated
    # at the power of 4
    x_avg = np.mean(x_avg ** 4)

    return _iroot(x_avg, 4)


def intensity_factor_ftp_score(X, ftp):
    """ Compute the intensity factor using the FTP

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    ftp : float
        Functional Threshold Power.

    Returns
    -------
    score: float
        Return the intensity factor.
    """

    # Check the conformity of X
    X = _check_X(X)

    # Compute the normalized power
    np_score = normalized_power_score(X)

    return np_score / ftp


def intensity_factor_mpa_score(X, mpa):
    """ Compute the intensity factor using the MPA

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    mpa : float
        Maximum Power Anaerobic.

    Returns
    -------
    score: float
        Return the intensity factor.
    """

    # Check the conformity of X
    X = _check_X(X)

    # Convert MPA to FTP
    ftp = 0.76 * mpa

    # Compute the resulting IF
    return intensity_factor_ftp_score(X, ftp)


def training_stress_ftp_score(X, ftp):
    """ Compute the training stress score using the FTP

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    ftp : float
        Functional Threshold Power.

    Returns
    -------
    score: float
        Return the intensity factor.

    """

    # Check the conformity of X
    X = _check_X(X)

    # Compute the intensity factor score
    if_score = intensity_factor_ftp_score(X, ftp)

    # Compute the training stress score
    return (X.size * if_score ** 2) / 3600.


def training_stress_mpa_score(X, mpa):
    """ Compute the training stress score

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    mpa : float
        Maximum Power Anaerobic.

    Returns
    -------
    score: float
        Return the intensity factor.
    """

    # Check the conformity of X
    X = _check_X(X)

    # Convert the mpa to ftp
    ftp = 0.76 * mpa

    # Compute the training stress score
    return training_stress_ftp_score(X, ftp)


def _iroot(k, n):
    """ Compute the root of something
    """
    u, s = n, n+1
    while u < s:
        s = u
        t = (k-1) * s + n // pow(s, k-1)
        u = t // k
    return s
