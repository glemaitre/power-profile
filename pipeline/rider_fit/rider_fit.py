import sys
import os

from skcycling.data_management import Rider


# The first argument corresponds to the root directory of a rider
path_data = sys.argv[1]

# The second folder is where to store the rider
store_path = sys.argv[2]

# The rider's parameters
max_duration = 300
cyclist_weight = None

# Create a rider
rider = Rider(max_duration=max_duration, cyclist_weight=cyclist_weight)

# Check all the sub-folder which correspond to the year
for dirs in os.walk('.').next()[1]:
    rider.add_rides_from_path(os.path.join(path_data, dirs))

# Save the rider
rider.save_to_pickles(store_path)
