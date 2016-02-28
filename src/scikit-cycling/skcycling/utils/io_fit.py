import numpy as np

from fitparse import FitFile

def load_power_from_fit(filename):
    """ Method to open the power data from FIT file into a numpy array.

    Parameters
    ----------
    filename: str,
        Path to the FIT file.
    """

    # Check that the filename has the good extension
    if filename.endswith('.fit') is not True:
        raise ValueError('The file does not have the right extension. Expected *.fit.')

    # Create an object to open the activity
    activity = FitFile(filename)
    activity.parse()

    # Get only the power records
    records = list(activity.get_messages(name='record'))

    # Append the different values inside a list which will be later
    # converted to numpy array
    power_rec = np.zeros((len(records), ))
    # Go through each record
    for idx_rec, rec in enumerate(records):
        # Extract only the value regarding the power
        p = rec.get_value('power')
        if p is not None:
            power_rec[idx_rec] = float(p)
        else:
            # raise ValueError('There record without power values. Check what is happening.')
            # We put the value to 0 since that it will not influence
            # the computation of the RPP
            power_rec[idx_rec] = 0.

    return power_rec
