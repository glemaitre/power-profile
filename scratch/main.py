import numpy as np
import matplotlib.pyplot as plt

from skcycling.utils import load_power_from_fit
from skcycling.power_profile import Rpp

# Define two files - We need to go through a folder automatically
ride_1 = '../data/user_1/2014/2014-05-12-09-31-05.fit'
ride_2 = '../data/user_1/2014/2014-05-29-08-12-03.fit'

# Extract the power from each file
power_ride_1 = load_power_from_fit(ride_1)
power_ride_2 = load_power_from_fit(ride_2)

# Create the object to handle the rider power-profile
# We will compute the rpp only on the 10 first minutes
max_duration_rpp = 10
rpp_rider = Rpp(max_duration_rpp=max_duration_rpp)

# Fit the first ride
rpp_rider.fit(power_ride_1)

# You want to see your actual profile
plt.plot(np.linspace(0, max_duration_rpp, max_duration_rpp * 60),
         rpp_rider.rpp_)

# We can update the profile either using fit or partial_fit
# However, partial_fit allows to override the profile if wanted
rpp_rider.fit(power_ride_2)

# You want to see your actual profile
plt.plot(np.linspace(0, max_duration_rpp, max_duration_rpp * 60),
         rpp_rider.rpp_)

# Show the plot
plt.show()
