import numpy as np 

def _moving_average(a, n=30) :
        """ Compute the normalized power for a given ride
            
        Parameters                 
        ----------                 
        a : array-like, shape (n_samples, )
        Array containing the ride or a selection of a ride.

        Returns
        -------
        ret : array-like (float)
        Return the denoising data mean-filter.
        """

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
