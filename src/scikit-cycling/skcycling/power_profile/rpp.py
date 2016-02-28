import numpy as np

from joblib import Parallel, delayed
import multiprocessing


def _rpp_parallel(self, X, idx_t_rpp):
    """ Function to compute the rpp in parallel

    Return
    ------
    power : float
        Returns the best power for the given duration of the rpp.
    """
    # Slice the data such that we can compute efficiently the mean later
    t_crop = np.array([X[i:-idx_t_rpp + i:]
                       for i in range(idx_t_rpp)])
    # Check that there is some value cropped. In the case that
    # the duration is longer than the file, the table crop is
    # empty
    if t_crop.size is not 0:
        # Compute the mean for each of these samples
        t_crop_mean = np.mean(t_crop, axis=0)
        # Keep the best to store as rpp
        return np.max(t_crop_mean)
    else:
        return 0


class Rpp(object):
    """ Rider power-profile

    Can perform online updates via `partial_fit` method.

    Parameters
    ----------
    max_duration_rpp : int
        Integer representing the maximum duration in minutes to
        build the rider power-profile model.

    Attributes
    ----------
    rpp_ : array-like, shape (60 * max_duration_rpp, )
        Array in which the rider power-profile is stored.
        The units used is the second.

    max_duration_rpp_ : int
        The maximum duration of the rider power-profile.
    """

    def __init__(self, max_duration_rpp):
        self.max_duration_rpp = max_duration_rpp


    def _check_partial_fit_first_call(self):
        """ Private function helper to know if the rider power-profile
            already has been computed.
        """

        # Check if the rider power-profile was previously created
        if getattr(self, 'rpp_', None) is not None:
            # Not the first time that the fitting is called
            return False
        else:
            # First time that the fitting is called
            # Initalise the rpp_ variable
            self.max_duration_rpp_ = self.max_duration_rpp
            self.rpp_ = np.zeros(60 * self.max_duration_rpp)

            return True

    @classmethod
    def _check_X(cls, X):
        """ Private function helper to check if X is of proper size

        Parameters
        ----------
        X : array-like, shape = (data, )
        """

        # Check that X is a numpy vector
        if len(X.shape) is not 1:
            raise ValueError('The shape of X is not consistent. It should be a 1D numpy vector.')
        # Check that X is of type float
        if X.dtype is not 'np.float64':
            X = X.astype(np.float64)

        return X

    def fit(self, X, in_parallel=True):
        """ Fit the data to the RPP

        Parameters
        ----------
        X : array-like, shape (n_samples, )

        in_parallel : boolean
            If True, the rpp will be computed on all the available cores

        Return
        ------
        self : object
            Returns self

        """

        # We should check if X is proper
        X = self._check_X(X)

        # Make a partial fitting of the current data
        return self.partial_fit(X, in_parallel=in_parallel)


    def partial_fit(self, X, refit=False, in_parallel=True):
        """ Incremental fit of the RPPB

        Parameters
        ----------
        X : array-like, shape (n_samples, )

        in_parallel : boolean
            If True, the rpp will be computed on all the available cores

        Return
        ------
        self : object
            Returns self

        """

        # Check that X is proper
        X = self._check_X(X)

        # Call the partial fitting
        return self._partial_fit(X, refit=refit, in_parallel=in_parallel)


    def _partial_fit(self, X, refit=False, in_parallel=True):
        """ Actual implementation of RPP calculation

        Parameters
        ----------
        X : array-like, shape (n_samples, )

        _refit : bool
            If True, the RPP will be overidden.

        in_parallel : boolean
            If True, the rpp will be computed on all the available cores

        Return
        ------
        self : object
            Returns self

        """

        # We should check X
        X = self._check_X(X)

        # If we want to recompute the rider power-profile
        if refit:
            self.rpp_ = None

        # Check if this the first called or if we have to initialize
        # the rider power-profile
        if self._check_partial_fit_first_call():
            # What to do if it was the first call
            # Compute the rider power-profile for the given X
            self.rpp_ = self._compute_ride_rpp(X, in_parallel)
        else:
            # What to do if it was yet another call
            # Compute the rider power-profile for the given X
            self.rpp = self._compute_ride_rpp(X, in_parallel)
            # Update the best rider power-profile
            self._update_rpp()

        return self

    def _compute_ride_rpp(self, X, in_parallel=True):
        """ Compute the rider power-profile

        Parameters
        ----------
        X : array-like, shape (n_samples, )

        in_parallel : boolean
            If True, the rpp will be computed on all the available cores

        Return
        ------
        rpp : array-like, shape (n_samples, )
            Array containing the rider power-profile of the current ride
        """

        # Check that X is proper
        X = self._check_X(X)

        if in_parallel is not True:
            # Initialize the ride rpp
            rpp = np.zeros(60 * self.max_duration_rpp_)

            # For each duration in the rpp
            for idx_t_rpp in range(rpp.size):
                # Slice the data such that we can compute efficiently
                # the mean later
                t_crop = np.array([X[i:-idx_t_rpp + i:]
                                   for i in range(idx_t_rpp)])
                # Check that there is some value cropped. In the case that
                # the duration is longer than the file, the table crop is
                # empty
                if t_crop.size is not 0:
                    # Compute the mean for each of these samples
                    t_crop_mean = np.mean(t_crop, axis=0)
                    # Keep the best to store as rpp
                    rpp[idx_t_rpp] = np.max(t_crop_mean)

            return rpp

        else:
            rpp = Parallel(n_jobs=-1)(delayed(_rpp_parallel)(self, X, idx_t_rpp)
                                      for idx_t_rpp
                                      in range(60 * self.max_duration_rpp_))
            # We need to make a conversion from list to numpy array
            return np.array(rpp)


    def _update_rpp(self):
        """ Update the rider power-profile
        """

        # Create a local copy of the best rpp
        b_rpp = self.rpp_.copy()

        # We have to compare the ride rpp with the best rpp
        for idx_rpp, (t_rpp, t_best_rpp) in enumerate(zip(self.rpp, self.rpp_)):
            # Update the best rpp in case the power is greater
            if t_rpp > t_best_rpp:
                b_rpp[idx_rpp] = t_rpp

        # Apply the copy
        self.rpp_ = b_rpp.copy()

        # In case that current rpp has an higher duration
        # we can append the value
        if len(self.rpp) > len(self.rpp_):
            # Update the max duration of the rpp
            self.rpp_ = np.append(self.rpp_, self.rpp[len(self.rpp_):])
            self.max_duration_rpp_ = int(len(self.rpp_ / 60.))
