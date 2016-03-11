""" Metrics to asses the performance of a cycling ride

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

from ..utils.checker import _check_X
from ..utils.running_average import _moving_average
import numpy as np

def normalized_power_score(X, pma):
    """ Compute the normalized power for a given ride

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.
    pma : double 
        Value 
    Returns
    -------
    score : float
        Return the normalized power.
    """

    # Check the conformity of X
    X = _check_X(X)
    x = _moving_average(X,n=30)
    
    # Removing value < 35% PMA
    
    arr = np.array([[-1, 5], [1, 1], [3, 11], [-4, 20], [2, 9]])
    X[~(X[:] < 0.35*pma)]
    
    X = np.power(X,4)
    X_mean = np.mean(X)
    
    np = _iroot(X_mean,4)
    

    return np


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

    # Compute the resulting IF
    np = normalized_power_score(X)

    if_score = np/ftp 
    return if_score 


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

    # Compute the resulting IF

    np = normalized_power_score(X)
    ftp = 0.76 * mpa  
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

    IF_val = intensity_factor_ftp_score(X,ftp)


    TSS = (np.size(X)*np.power(IF_val, 2))(3600)


    return


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
    
    ftp = 0.76*mpa 



    return training_stress_ftp_score(X, ftp)

    def _iroot(k, n):
        u, s = n, n+1
        while u < s:
            s = u
            t = (k-1) * s + n // pow(s, k-1)
            u = t // k
        return s


