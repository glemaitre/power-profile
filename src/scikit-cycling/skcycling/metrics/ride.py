""" Metrics to asses the performance of a cycling ride

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

from ..utils.checker import _check_X
from ..utils.checker import _check_float

from ..restoration.denoise import moving_average

import numpy as np

TS_SCALE_GRAPPE = dict([('I1', 2.), ('I2', 2.5), ('I3', 3.),
                        ('I4', 3.5), ('I5', 4.5), ('I6', 7.),
                        ('I7', 11.)])

ESIE_SCALE_GRAPPE = dict([('I1', (.3, .5)), ('I2', (.5, .6)),
                          ('I3', (.6, .75)), ('I4', (.75, .85)),
                          ('I5', (.85, 1.)), ('I6', (1., 1.80)),
                          ('I7', (1.8, 3.))])


def normalized_power_score(X, pma):
    """ Compute the normalized power for a given ride

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    pma : float
        Maxixum Anaerobic Power.

    Returns
    -------
    score : float
        Return the normalized power.
    """

    # Check the conformity of X and pma
    X = _check_X(X)
    pma = _check_float(pma)

    # Denoise the rpp through moving average using 30 sec filter
    x_avg = moving_average(X, win=30)

    # Removing value < I1-ESIE, i.e. 30 % PMA
    x_avg = np.delete(x_avg, np.nonzero(x_avg <
                                        (ESIE_SCALE_GRAPPE['I1'][0] * pma)))

    # Compute the mean of the denoised ride elevated
    # at the power of 4
    return np.mean(x_avg ** 4.) ** (1. / 4.)


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

    # Check the conformity of X and ftp
    X = _check_X(X)
    ftp = _check_float(ftp)

    # Compute the normalized power
    np_score = normalized_power_score(X, ftp2pma(ftp))

    return np_score / ftp


def intensity_factor_pma_score(X, pma):
    """ Compute the intensity factor using the PMAB

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    pma : float
        Maximum Anaerobic Power.

    Returns
    -------
    score: float
        Return the intensity factor.
    """

    # Check the conformity of X and pma
    X = _check_X(X)
    pma = _check_float(pma)

    # Compute the resulting IF
    return intensity_factor_ftp_score(X, pma2ftp(pma))


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
        Return the training stress score.

    """

    # Check the conformity of X and ftp
    X = _check_X(X)
    ftp = _check_float(ftp)

    # Compute the intensity factor score
    if_score = intensity_factor_ftp_score(X, ftp)

    # Compute the training stress score
    return (X.size * if_score ** 2) / 3600.


def training_stress_pma_score(X, pma):
    """ Compute the training stress score

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    pma : float
        Maximum Anaerobic Power.

    Returns
    -------
    score: float
        Return the training stress score.
    """

    # Check the conformity of X and pma
    X = _check_X(X)
    pma = _check_float(pma)

    # Compute the training stress score
    return training_stress_ftp_score(X, pma2ftp(pma))


def pma2ftp(pma):
    """ Convert the PMA to FTP

    Parameter:
    ----------
    pma : float
        Maximum Anaerobic Power.

    Return:
    -------
    ftp : float
        Functioning Threhold Power.
    """

    return 0.76 * pma


def ftp2pma(ftp):
    """ Convert the PMA to FTP

    Parameter:
    ----------
    ftp : float
        Functioning Threhold Power.

    Return:
    -------
    pma : float
        Maximum Anaerobic Power.
    """

    return ftp / 0.76


def training_stress_pma_grappe_score(X, pma):
    """ Compute the training stress score using the PMA,
        considering Grappe et al. approach

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    pma : float
        Maximum Anaerobic Power.

    Returns
    -------
    tss_score: float
        Return the training stress score.
    """

    # Check the consistency of X and pma
    X = _check_X(X)
    pma = _check_float(pma)

    # Compute the stress for each item of the ESIE
    tss_grappe = 0.
    for key_sc in TS_SCALE_GRAPPE.iterkeys():

        # Count the number of elements which corresponds to as sec
        # We need to convert it to minutes
        curr_stress = np.count_nonzero(
            np.bitwise_and(X >= ESIE_SCALE_GRAPPE[key_sc][0] * pma,
                           X < ESIE_SCALE_GRAPPE[key_sc][1] * pma)) / 60.

        # Compute the cumulative stress
        tss_grappe += curr_stress * TS_SCALE_GRAPPE[key_sc]

    return tss_grappe


def training_stress_ftp_grappe_score(X, ftp):
    """ Compute the training stress score using the FTP,
        considering Grappe et al. approach

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    ftp : float
        Functional Threshold Power.

    Returns
    -------
    score: float
        Return the training stress score.

    """

    return training_stress_pma_grappe_score(X, ftp2pma(ftp))
