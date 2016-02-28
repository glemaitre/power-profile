import numpy as np

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


    def fit(self, X):
        """ Fit the data to the RPP

        Parameters:
        -----------
        X : array-like, shape (n_samples, )

        """

        # We should check if X is proper

        # Make a partial fitting of the current data
        return self.partial_fit(self, X)


    def partial_fit(self, X, refit=False):
        """ Incremental fit of the RPPB

        Parameters:
        -----------
        X : array-like, shape (n_samples, )

        """

        return self._partial_fit(self, X, refit=refit)


    def _partial_fit(self, X, refit=False):
        """ Actual implementation of RPP calculation

        Parameters:
        -----------
        X : array-like, shape (n_samples, )

        _refit : bool
            If True, the RPP will be overidden.

        """

        # We should check X

        # If we want to recompute the rider power-profile
        if refit:
            self.rpp_ = None

        # Check if this the first called or if we have to initialize
        # the rider power-profile
        if self._check_partial_fit_first_call():
            # What to do if it was the first call
            # Compute the rider power-profile for the given X
            self.rpp_ = self._compute_ride_rpp(X)
        else:
            # What to do if it was yet another call
            # Compute the rider power-profile for the given X
            self.rpp = self.compute_ride_rpp(X)
            # Update the best rider power-profile
            self._update_rpp()

        return self

    @staticmethod
    def _compute_ride_rpp(self, X):
        """ Compute the rider power-profile

        Parameters:
        -----------
        X : array-like, shape (n_samples, )
        """

        # Initialize the ride rpp
        rpp = np.zeros(60 * self.max_duration_rpp_)

        # For each duration in the rpp
        for idx_t_rpp, t_rpp in enumerate(rpp):
            # Slice the data such that we can compute efficiently the mean later
            t_crop = np.array([X[i:-idx_t_rpp + i:]
                                for i in range(idx_t_rpp)])
            # Compute the mean for each of these samples
            t_crop_mean = np.mean(t_crop, axis=0)
            # Keep the best to store as rpp
            t_rpp = np.max(t_crop_mean)

        return rpp

    def _update_rpp(self):
        """ Update the rider power-profile
        """
        # We have to compare the ride rpp with the best rpp
        for t_rpp, t_best_rpp in zip(self.rpp, self.rpp_):
            # Update the best rpp in case the power is greater
            if t_rpp > t_best_rpp:
                t_best_rpp = t_rpp

    # def compute_rpp(self, data_ex, mode=0, existing_rpp=[]):
    #     d_rpp = np.size(self.duration_rpp)
    #     d_ex = np.size(data_ex)
    #     t_crop = []
    #     t_mean_slip = []

    #     for i in range(d_rpp):
    #         # pour toutes les durées
    #         for j in range(d_ex-i):
    #             # pour tous les élémerts du tableau
    #             t_crop = data_ex[j:j+i]
    #             #crop de la portion du tableau
    #             t_mean_slip.append(np.mean(t_crop))
    #             t_crop = []
    #             # calculer toutes les moyennes glissantes de tailles
    #         self.res_rpp[i] = np.max(t_mean_slip)
    #         t_mean_slip = []
            
    #         if mode == 0: # si mode pas de mise à jours rpp existant
    #             return self.res_rpp
    #         else: # si mode mise à jours rpp existant
    #             size_existing_rpp = np.size(existing_rpp)
    #             if size_existing_rpp == self.duration_rpp : # taille rpp existant égale  à taille rppp calculé
    #                 for i in range(size_existing_rpp):
    #                     if self.res_rpp[i] > existing_rpp:
    #                         existing_rpp[i] = self.res_rpp[i]

    #             if size_existing_rpp > self.res_rpp: # si existant plus grand que le rpp calculé
    #                 for i in range(size_existing_rpp):
    #                     if self.res_rpp[i] > existing_rpp:
    #                         existing_rpp[i] = self.res_rpp[i]


    #             if size_existing_rpp < self.res_rpp: # si exsitant plus petit que le rpp calculé
    #                 for i in range(size_existing_rpp): # partie commune
    #                     if self.res_rpp[i] > existing_rpp:
    #                         existing_rpp[i] = self.res_rpp[i] # partie supplémentaire
    #                 for i in range(size_existing_rpp+1, self.duration_rpp):
    #                     existing_rpp.append(self.res_rpp[i])
                
    #             return existing_rpp 

    # def get_res_rpp(self):

    #     return self.res_rpp

    # def get_duration_rpp(self):

    #     return self.duration_rpp
