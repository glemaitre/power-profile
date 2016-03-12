""" Testing the metrics developed to asses performance of a ride """

import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.metrics import normalized_power_score
from skcycling.metrics import intensity_factor_ftp_score
from skcycling.metrics import intensity_factor_mpa_score
from skcycling.metrics import training_stress_ftp_score
from skcycling.metrics import training_stress_mpa_score
