import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from skcycling.power_profile import Rpp

# Load some data previously acquired
path_to_profile = '../../data/user_1/profile/2014/profile.npy'
rider_rpp = Rpp.load_from_npy(path_to_profile, cyclist_weight=70.)

# plt.figure()

# # Plot the original data
# plt.plot(np.linspace(0, rider_rpp.max_duration_rpp_, rider_rpp.rpp_.size), 
#          rider_rpp.rpp_norm_)

# # # Plot a denoised signal using b-spline
# # plt.plot(np.linspace(0, rider_rpp.max_duration_rpp_, rider_rpp.rpp_.size),
# #          rider_rpp.denoise_rpp(normalized=True), 'r+')

# # Resample the data as the SRM
ts = np.array([0.016, 0.083, 0.5, 1, 3, 3.5, 4, 4.5, 5, 5.5,
               6, 6.5, 7, 10, 20, 30, 45, 60, 120, 180, 240])
# plt.plot(ts, rider_rpp.resampling_rpp(ts, normalized=True))

# # Show the plot
# plt.show()

# Fit the model from Pinot et al.
ts_reg = np.array([4, 4.5, 5, 5.5,
                   6, 6.5, 7, 10, 20, 30,
                   45, 60, 120, 180, 240])
slope, intercept, std_err, r_squared = rider_rpp.aerobic_meta_model(ts=ts_reg,
                                                                    normalized=True,
                                                                    method='lsq')

print 'The goodness of fitting is R2: {}'.format(r_squared)

# Define a lambda function to plot a line
def line_generator(x, slope, intercept):
    return slope * x + intercept

plt.figure()

# Plot the specific point in a semilog plot
plt.semilogx(ts, rider_rpp.resampling_rpp(ts, normalized=True), 'ro')
plt.semilogx(ts, line_generator(np.log(ts), slope, intercept))
plt.fill_between(ts, 
                 line_generator(np.log(ts), slope,
                                intercept) + 2 * std_err,
                 line_generator(np.log(ts), slope,
                                intercept) - 2 * std_err,
                 alpha=0.2)

# Show the plot
plt.show()
