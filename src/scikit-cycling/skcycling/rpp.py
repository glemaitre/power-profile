import numpy as np

class Rpp(object):
    """ Class to power profile """

    def __init__(self, duration):

        self.duration_rpp = duration
        self.res_rpp = np.zeros(60*self.duration)

    def compute_rpp(self, data_ex, mode=0, existing_rpp=[]):
        d_rpp = np.size(self.duration_rpp)
        d_ex = np.size(data_ex)
        t_crop = []
        t_mean_slip = []

        for i in range(d_rpp):
            # pour toutes les durées
            for j in range(d_ex-i):
                # pour tous les élémerts du tableau
                t_crop = data_ex[j:j+i]
                #crop de la portion du tableau
                t_mean_slip.append(np.mean(t_crop))
                t_crop = []
                # calculer toutes les moyennes glissantes de tailles
            self.res_rpp[i] = np.max(t_mean_slip)
            t_mean_slip = []
            
            if mode == 0: # si mode pas de mise à jours rpp existant
                return self.res_rpp
            else: # si mode mise à jours rpp existant
                size_existing_rpp = np.size(existing_rpp)
                if size_existing_rpp == self.duration_rpp : # taille rpp existant égale  à taille rppp calculé
                    for i in range(size_existing_rpp):
                        if self.res_rpp[i] > existing_rpp:
                            existing_rpp[i] = self.res_rpp[i]

                if size_existing_rpp > self.res_rpp: # si existant plus grand que le rpp calculé
                    for i in range(size_existing_rpp):
                        if self.res_rpp[i] > existing_rpp:
                            existing_rpp[i] = self.res_rpp[i]


                if size_existing_rpp < self.res_rpp: # si exsitant plus petit que le rpp calculé
                    for i in range(size_existing_rpp): # partie commune
                        if self.res_rpp[i] > existing_rpp:
                            existing_rpp[i] = self.res_rpp[i] # partie supplémentaire
                    for i in range(size_existing_rpp+1, self.duration_rpp):
                        existing_rpp.append(self.res_rpp[i])
                
                return existing_rpp 

    def get_res_rpp(self):

        return self.res_rpp

    def get_duration_rpp(self):

        return self.duration_rpp

    def fit(self, X):
        """ Fit the data to the RPP
        
        Parameters:
        -----------
        X : array-like, shape (n_samples)

        """

        # We should check if X is proper
        
        # Make a partial fitting of the current data
        return self.partial_fit(X)

    def partial_fit(self, X):
        """ Incremental fit of the RPPB
        
        Parameters:
        -----------
        X : array-like, shape (n_samples)

        """

        return self._partial_fit(self, X)

    def _partial_fit(self, X, _refit=False):
        """ Actual implementation of RPP calculation
        
        Parameters:
        -----------
        X : array-like, shape (n_samples)

        _refit : bool
            If True, the RPP will be overidden.

        """

        # We should check X

        
