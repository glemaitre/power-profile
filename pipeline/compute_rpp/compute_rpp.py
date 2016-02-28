import sys
import os
import numpy as np

from skcycling.utils import load_power_from_fit
from skcycling.power_profile import Rpp

# The first input argument corresponding to the data path
data_path = sys.argv[1]

# The second input argument is the storage directory
storage_path = sys.argv[2]

# We can create a list of all the *.fit files present inside that directory
# Create a list with the files to considered
filenames = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith('.fit'):
            filenames.append(os.path.join(root, file))

max_duration_rpp = 300
rpp_rider = Rpp(max_duration_rpp=max_duration_rpp)
# Open each file and fit
for filename in filenames:
    # Open the file
    power_ride = load_power_from_fit(filename)
    # Fit the ride
    rpp_rider.fit(power_ride)

# Create a directory to store the data if it is not existing
if not os.path.exists(storage_path):
    os.makedirs(storage_path)

# Store the data somewhere
np.save(os.path.join(storage_path, 'profile.npy'), rpp_rider.rpp_)
