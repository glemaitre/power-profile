import numpy as np
import matplotlib.pyplot as plt

from skcycling.power_profile import Rpp

# Load some data previously acquired
path_to_profile = '../../data/user_1/profile/2015/profile.npy'
rider_rpp = Rpp.load_from_npy(path_to_profile)

# Plot the original data
plt.plot(np.linspace(0, rider_rpp.max_duration_rpp_, rider_rpp.rpp_.size), 
         rider_rpp.rpp_)

# Plot a denoised signal using b-spline
plt.plot(np.linspace(0, rider_rpp.max_duration_rpp_, rider_rpp.rpp_.size), 
         rider_rpp.denoise_rpp())

# Resample the data as the SRM
ts = np.array([0.016, 0.083, 0.5, 1, 3, 3.5, 4, 4.5, 5, 5.5,
               6, 6.5, 7, 10, 20, 30])
plt.plot(ts, rider_rpp.resampling_rpp(ts))

# Show the plot
plt.show()
