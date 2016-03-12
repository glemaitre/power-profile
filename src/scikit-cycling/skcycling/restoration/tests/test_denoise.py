import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.restoration import outliers_rejection
from skcycling.restoration import moving_average

pow_ride_1 = np.linspace(-10., 3000., 20)
pow_ride_2 = np.array([200.]*4 + [300]*4)


def test_outliers_thres_rejection():
    """ Test the outlier rejection method based on thresholding """
    # Declare the table for later comparison
    X_comp = np.array([1495., 148.42105263, 306.84210526, 465.26315789,
                      623.68421053, 782.10526316, 940.52631579,
                      1098.94736842, 1257.36842105, 1415.78947368,
                      1574.21052632, 1732.63157895, 1891.05263158,
                      2049.47368421, 2207.89473684, 2366.31578947,
                      1495., 1495., 1495., 1495.])

    # Make the outliers rejection
    X_free_outliers = outliers_rejection(pow_ride_1)

    # Check if they are the same
    assert_array_almost_equal(X_free_outliers, X_comp)


def test_moving_average():
    """ Test the moving average """

    # Declare the ground-truth table
    X_gt = np.array([200., 200., 233.333333, 266.666667, 300., 300.])

    # Denoise the signal using moving average
    X_denoise = moving_average(pow_ride_2, win=3)

    assert_array_almost_equal(X_denoise, X_gt)
