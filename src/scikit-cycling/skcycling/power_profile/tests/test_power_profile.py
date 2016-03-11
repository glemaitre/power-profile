import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.power_profile import Rpp

pow_ride_1 = np.linspace(100, 200, 4000)
pow_ride_2 = np.linspace(50, 400, 4000)


def test_fitting_rpp():
    """ Test the computation of the fitting """
    my_rpp = Rpp(max_duration_rpp=10)
    my_rpp.fit(pow_ride_1)

    # We need to make an assert equal here with meaningful data
    print my_rpp.rpp_


def test_partial_fitting_rpp():
    """ Test the partial fitting with update of the rpp """
    my_rpp = Rpp(max_duration_rpp=10)
    my_rpp.fit(pow_ride_1)
    my_rpp.partial_fit(pow_ride_2)

    # We need to make an assert equal here with meaningful data
    print my_rpp.rpp_


def test_partial_fitting_rpp_refit():
    """ Test the partial fitting with update of the rpp """
    my_rpp = Rpp(max_duration_rpp=10)
    my_rpp.fit(pow_ride_1)
    my_rpp.partial_fit(pow_ride_2)
    my_rpp.partial_fit(pow_ride_1, refit=True)

    # We need to make an assert equal here with meaningful data
    print my_rpp.rpp_
