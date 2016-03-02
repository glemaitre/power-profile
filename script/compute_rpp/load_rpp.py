import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from skcycling.power_profile import Rpp

# Load some data previously acquired
path_to_profile = '../../data/user_1/profile/2015/profile.npy'
rider_rpp = Rpp.load_from_npy(path_to_profile)

plt.figure()

# Plot the original data
plt.plot(np.linspace(0, rider_rpp.max_duration_rpp_, rider_rpp.rpp_.size), 
         rider_rpp.rpp_)

# Plot a denoised signal using b-spline
plt.plot(np.linspace(0, rider_rpp.max_duration_rpp_, rider_rpp.rpp_.size), 
         rider_rpp.denoise_rpp())

# Resample the data as the SRM
ts = np.array([3, 3.5, 4, 4.5, 5, 5.5,
               6, 6.5, 7, 10, 20, 30])
plt.plot(ts, rider_rpp.resampling_rpp(ts))

# Show the plot
plt.show()

# Fit the model from Pinot et al.
fit_info = rider_rpp.aerobic_meta_model()

# Define a lambda function to plot a line
def line_generator(x, slope, intercept):
    return slope * x + intercept

plt.figure()

# Plot the specific point in a semilog plot
plt.plot(np.log(ts), rider_rpp.resampling_rpp(ts), 'ro')
plt.plot(np.log(ts), line_generator(np.log(ts), fit_info.slope, fit_info.intercept))
# plt.fill_between(np.log(ts), 
#                  line_generator(np.log(ts), fit_info.slope,
#                                 fit_info.intercept) + 2 * fit_info.stderr,
#                  line_generator(np.log(ts), fit_info.slope,
#                                 fit_info.intercept) - 2 * fit_info.stderr,
#                  alpha=0.2)

# Show the plot
plt.show()
