import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from joblib import Parallel, delayed
import multiprocessing


from ..utils.checker import _check_X


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
    """ Record power-profile

    Can perform online updates via `partial_fit` method.

    Parameters
    ----------
    max_duration_rpp : int
        Integer representing the maximum duration in minutes to
        build the record power-profile model.

    cyclist_weight : float, default None
        Float in order to normalise the record power-profile depending
        of its weight. By default this is None in order to avoid
        using the data from normalized rpp without this data.

    Attributes
    ----------
    rpp_ : array-like, shape (60 * max_duration_rpp, )
        Array in which the record power-profile is stored.
        The units used is the second.

    rpp_norm_ : array-like, shape (60 * max_duration_rpp, )
        Array in which the weight-normalized record power-profile
        is stored. The units used is the seconds.

    max_duration_rpp_ : int
        The maximum duration of the record power-profile.

    cyclist_weight_ : float
        Cyclist weight.
    """

    def __init__(self, max_duration_rpp, cyclist_weight=None):
        self.max_duration_rpp = max_duration_rpp
        self.cyclist_weight = cyclist_weight

    def _check_partial_fit_first_call(self):
        """ Private function helper to know if the record power-profile
            already has been computed.
        """

        # Check if the record power-profile was previously created
        if getattr(self, 'rpp_', None) is not None:
            # Not the first time that the fitting is called
            return False
        else:
            # First time that the fitting is called
            # Initalise the rpp_ variable
            self.max_duration_rpp_ = self.max_duration_rpp
            self.cyclist_weight_ = self.cyclist_weight
            self.rpp_ = np.zeros(60 * self.max_duration_rpp)
            # If the weight is not None we can also initialize the
            # normalized rpp
            if self.cyclist_weight_ is not None:
                self.rpp_norm_ = self.rpp_
            else:
                self.rpp_norm_ = None

            return True

    @classmethod
    def load_from_npy(cls, filename, cyclist_weight=None):
        """ Load the record power-profile from an npy file

        Parameters
        ----------
        filename : str
            String containing the path to the NPY file containing the array
            representing the record power-profile.

        cyclist_weight : float, default None
            Float in order to normalise the record power-profile depending
            of its weight. By default this is None in order to avoid
            using the data from normalized rpp without this data.

        Return
        ------
        self : object
            Returns self
        """

        # Load the record power-profile
        cls.rpp_ = np.load(filename)

        # We have to infer the duration of the rpp
        max_duration_rpp = cls.rpp_.size / 60
        cls.max_duration_rpp_ = max_duration_rpp

        # Apply the cyclist weight
        cls.cyclist_weight_ = cyclist_weight

        # Compute the normalized rpp if possible
        if cls.cyclist_weight_ is not None:
            cls.rpp_norm_ = cls.rpp_ / cls.cyclist_weight_
        else:
            cls.rpp_norm_ = None

        return cls(max_duration_rpp)

    def fit(self, X, in_parallel=True):
        """ Fit the data to the RPP

        Parameters
        ----------
        X : array-like, shape (n_samples, )

        in_parallel : boolean
            If True, the rpp will be computed on all the available cores.

        Return
        ------
        self : object
            Returns self.

        """

        # We should check if X is proper
        X = _check_X(X)

        # Make a partial fitting of the current data
        return self.partial_fit(X, refit=False, in_parallel=in_parallel)

    def partial_fit(self, X, refit=False, in_parallel=True):
        """ Incremental fit of the RPPB

        Parameters
        ----------
        X : array-like, shape (n_samples, )

        in_parallel : boolean
            If True, the rpp will be computed on all the available cores.

        Return
        ------
        self : object
            Returns self.

        """

        # Check that X is proper
        X = _check_X(X)

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
            If True, the rpp will be computed on all the available cores.

        Return
        ------
        self : object
            Returns self.

        """

        # We should check X
        X = _check_X(X)

        # If we want to recompute the record power-profile
        if refit:
            self.rpp_ = None
            self.rpp_norm_ = None

        # Check if this the first called or if we have to initialize
        # the record power-profile
        if self._check_partial_fit_first_call():
            # What to do if it was the first call
            # Compute the record power-profile for the given X
            self.rpp_ = self._compute_ride_rpp(X,
                                               self.max_duration_rpp_,
                                               in_parallel)
            # Compute the normalized rpp if we should
            if self.cyclist_weight_ is not None:
                self.rpp_norm_ = self.rpp_ / self.cyclist_weight_
        else:
            # What to do if it was yet another call
            # Compute the record power-profile for the given X
            self.rpp = self._compute_ride_rpp(X,
                                              self.max_duration_rpp_,
                                              in_parallel)
            # Update the best record power-profile
            self._update_rpp()
            # Compute the normalized rpp if we should
            if self.cyclist_weight_ is not None:
                self.rpp_norm_ = self.rpp_ / self.cyclist_weight_

        return self

    @classmethod
    def _compute_ride_rpp(cls, X, max_duration_rpp, in_parallel=True):
        """ Compute the record power-profile

        Parameters
        ----------
        X : array-like, shape (n_samples, )

        in_parallel : boolean
            If True, the rpp will be computed on all the available cores.

        Return
        ------
        rpp : array-like, shape (n_samples, )
            Array containing the record power-profile of the current ride.
        """

        # Check that X is proper
        X = _check_X(X)

        if in_parallel is not True:
            # Initialize the ride rpp
            rpp = np.zeros(60 * max_duration_rpp)

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
            rpp = Parallel(n_jobs=-1)(delayed(_rpp_parallel)(cls, X, idx_t_rpp)
                                      for idx_t_rpp
                                      in range(60 * max_duration_rpp))
            # We need to make a conversion from list to numpy array
            return np.array(rpp)

    def _update_rpp(self):
        """ Update the record power-profile
        """

        # Create a local copy of the best rpp
        b_rpp = self.rpp_.copy()

        # We have to compare the ride rpp with the best rpp
        for idx_rpp, (t_rpp, t_best_rpp) in enumerate(zip(self.rpp,
                                                          self.rpp_)):
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

    def denoise_rpp(self, method='b-spline', normalized=False):
        """ Denoise the record power-profile

        Parameters
        ----------
        method : str, default 'b-spline'
            Method to select to denoise the record power-profile.

        normalized : bool, default False
            Return a weight-normalized rpp if True.

        Return
        ------
        rpp : array-like, shape (n_samples, )
            Return a denoise record power-profile.
        """

        if method == 'b-spline':
            # Shall used the rpp or weight-normalized rpp
            if normalized is True:
                # Check that the cyclist weight was provided
                if self.cyclist_weight_ is not None:
                    rpp = self.rpp_norm_
                else:
                    raise ValueError('You cannot get a normalized rpp if the'
                                     ' cyclist weight never has been given.')
            else:
                rpp = self.rpp_

            # Apply denoising based on b-spline
            # Create the timeline
            t = np.linspace(0, self.max_duration_rpp_, rpp.size)
            spl = UnivariateSpline(t, rpp)

            return spl(t)
        else:
            raise ValueError('This denoising method is not implemented.')

    def resampling_rpp(self, ts, method_interp='linear', normalized=False):
        """ Resampling the record power-profile

        Parameters
        ----------
        ts : array-like, shape (n_sample, )
            An array containaining the time landmark to sample.

        method_interp : string, default 'linear'
            Name of the method to interpolate the data.
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
            where 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of first, second or third order) or as an integer
            specifying the order of the spline interpolator to use.

        normalized : bool, default False
            Return a weight-normalized rpp if True.


        Return
        ------
        rpp : array-like, shape (n_samples, )
            Return a resampled record power-profile.
        """

        # Shall used the rpp or weight-normalized rpp
        if normalized is True:
            # Check that the cyclist weight was provided
            if self.cyclist_weight_ is not None:
                rpp = self.rpp_norm_
            else:
                raise ValueError('You cannot get a normalized rpp if the'
                                 ' cyclist weight never has been given.')
        else:
            rpp = self.rpp_

        t = np.linspace(0, self.max_duration_rpp_, rpp.size)
        f = interp1d(t, rpp, kind=method_interp)

        return f(ts)

    @staticmethod
    def _res_std_dev(model, estimate):
        """ Private function to compute the residual standard deviation

        Parameters
        ----------
        model : array-like, shape (n_sample, )
             Value used to made the fitting.

        estimate : array-like, shape (n_sample, )
             Value obtained by fitting.

        Return
        ------
        residual : float
            Residual standard deviation.
        """

        return np.sqrt(np.sum((model - estimate) ** 2) /
                       (float(model.size) - 2.))

    @staticmethod
    def _r_squared(model, estimate):
        """ Private function to compute the coefficient of determination

        Parameters
        ----------
        model : array-like, shape (n_sample, )
             Value used to made the fitting.

        estimate : array-like, shape (n_sample, )
             Value obtained by fitting.

        Return
        ------
        coeff_det : float
            Coefficient of determination.
        """

        # Compute of the observed data
        model_mean = np.mean(model)

        # Compute the total sum of squares
        ss_tot = np.sum((model - model_mean) ** 2)

        # Compute the sum of squares residual
        ss_res = np.sum((model - estimate) ** 2)

        return 1. - (ss_res / ss_tot)

    def aerobic_meta_model(self, ts=None, starting_time=4,
                           normalized=False, method='lsq'):
        """ Compute the aerobic metabolism model from the
            record power-profile

        Parameters
        ----------
        ts : array-like, shape (n_samples, )
            Array containing the sample to take into account.
            None if we want to pick up all the data.

        start_time : int, default 4
            Starting time to consider when fitting the linear model.

        normalized : bool, default False
            Return a weight-normalized rpp if True.

        method : string, default 'lsq'
            Which type of tehcnic to use to make the fitting ('lsq', 'lm').

        Return
        ------
        slope : float
            slope of the regression line.

        intercept : float
            intercept of the regression line.

        stderr : float
            Standard error of the estimate.

        coeff_det : float
            Coefficient of determination.

        Notes
        -----
        [1] Pinot et al., "Determination of Maximal Aerobic Power
        on the Field in Cylcing" (2014)

        """

        # Define the function to fit
        def linear_model(x, a, b):
            return a * x + b

        # If ts is not provided we have to create a timeline
        if ts is None:
            ts = np.linspace(starting_time,
                             self.max_duration_rpp_,
                             (self.max_duration_rpp_ - starting_time) * 60)

        # Compute the rpp
        rpp = self.resampling_rpp(ts, normalized=normalized)

        # The zero values need to be avoided for the fitting
        # Keep the signal which is not zero
        ts = ts[np.nonzero(rpp)]
        rpp = rpp[np.nonzero(rpp)]

        if method == 'lsq':
            # Perform the fitting using least-square
            slope, intercept, _, _, _ = linregress(np.log(ts), rpp)

            std_err = self._res_std_dev(rpp, linear_model(np.log(ts),
                                                          slope,
                                                          intercept))

            coeff_det = self._r_squared(rpp, linear_model(np.log(ts),
                                                          slope,
                                                          intercept))
        elif method == 'lm':
            # Perform the fitting using non-linear least-square
            # Levenberg-Marquardt
            popt, pcov = curve_fit(linear_model, np.log(ts), rpp)

            slope = popt[0]
            intercept = popt[1]

            std_err = self._res_std_dev(rpp, linear_model(np.log(ts),
                                                          slope,
                                                          intercept))

            coeff_det = self._r_squared(rpp, linear_model(np.log(ts),
                                                          slope,
                                                          intercept))

        return slope, intercept, std_err, coeff_det
