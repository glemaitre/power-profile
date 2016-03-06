""" Metrics to asses the performance of a cycling ride

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

from ..utils.checker import _check_X


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

    return


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

    return


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

    return


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

    return
