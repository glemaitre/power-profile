import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.restoration import denoise

pow_ride_1 = np.linspace(-10., 3000., 20)


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
    X_free_outliers = denoise.outliers_rejection(pow_ride_1)

    # Check if they are the same
    assert_array_almost_equal(X_free_outliers, X_comp)
