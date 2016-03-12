""" Testing the metrics developed to asses performance of a ride """

import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.metrics import normalized_power_score
from skcycling.metrics import intensity_factor_ftp_score
from skcycling.metrics import intensity_factor_pma_score
from skcycling.metrics import training_stress_ftp_score
from skcycling.metrics import training_stress_pma_score
from skcycling.metrics import pma2ftp
from skcycling.metrics import ftp2pma

pma = 400.
ftp = 304.
ride = np.array([300.]*200 + [0.]*200 + [200.]*200)


def test_normalized_power_score():
    """ Testing the function computing the NP """

    # Compute the NP
    np_score_gt = 261.25956394585199
    np_score = normalized_power_score(ride, pma)
    assert_almost_equal(np_score, np_score_gt)


def test_intensity_factor_ftp_score():
    """ Testing the function computing IF with FTP """

    if_score_ftp_gt = 0.85940646034819734
    if_score_ftp = intensity_factor_ftp_score(ride, ftp)
    assert_almost_equal(if_score_ftp, if_score_ftp_gt)


def test_intensity_factor_pma_score():
    """ Testing the function computing IF with PMA """

    if_score_pma_gt = 0.85940646034819734
    if_score_pma = intensity_factor_pma_score(ride, pma)
    assert_almost_equal(if_score_pma, if_score_pma_gt)


def test_training_stress_ftp_score():
    """ Testing the function to compute the TSS from FTP """

    tss_score_ftp_gt = 0.12309657734803628
    tss_score_ftp = training_stress_ftp_score(ride, ftp)
    assert_almost_equal(tss_score_ftp, tss_score_ftp_gt)


def test_training_stress_pma_score():
    """ Testing the function to compute the TSS from PMA """

    tss_score_pma_gt = 0.12309657734803628
    tss_score_pma = training_stress_pma_score(ride, pma)
    assert_almost_equal(tss_score_pma, tss_score_pma_gt)


def test_pma2ftp():
    """ Testing the function converting the PMA to FTP """

    ftp_score = pma2ftp(pma)
    assert_equal(ftp_score, ftp)


def test_ftp2pma():
    """ Testing the function converting the FTP to PMA """

    pma_score = ftp2pma(ftp)
    assert_equal(pma_score, pma)
