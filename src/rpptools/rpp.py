import numpy as np

class Rpp:

    def __init__(self, duration):
        """ Constructeur"""

        self.duration_rpp = duration
        self.res_rpp = np.zeros(60*self.duration)

    def compute_rpp(self, data_ex):
        d_rpp = np.size(self.duration_rpp)
        d_ex = np.size(data_ex)
        t_crop = []
        t_mean_slip = []

        for i in range(d_rpp):
            # pour toutes les durées
            for j in range(d_ex):
                # pour tous les élémerts du tableau
                t_crop = data_ex[j:j+1]
                #crop de la portion du tableau
                t_mean_slip.append(np.mean(t_crop))
                t_crop = []
                # calculer toutes les moyennes glissantes de tailles
            self.res_rpp[i] = np.max(t_mean_slip)
            t_mean_slip = []

    def get_res_rpp(self):

        return self.res_rpp

    def get_duration_rpp(self):

        return self.duration_rpp







