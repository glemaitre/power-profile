import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


def first_derivative(f) :
    taille = np.size(f)

    d_rpp = []
    for i in range(taille-1):
        d_rpp.append(f[i+1]-f[i])

    return d_rpp



def second_derivative(f) :
    d_rpp = first_derivative(f)
    dd_rpp = first_derivative(d_rpp)

    return dd_rpp 


rpp = np.load('profile.npy')
rpp_zoom = rpp[180:420]

#echantillonnage comme F Grappe et SRM
val_ech_srm =[1, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 10, 20, 29.9]
#val_ech_srm =[3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
tab_srm = []
size_tab_srm = np.size(val_ech_srm)

#copie

for i in range(size_tab_srm):
    ind = int(val_ech_srm[i]*60)
    tab_srm.append(rpp[ind])
dd_tb_srm = second_derivative(tab_srm)

dd_rpp_zoom = second_derivative(rpp_zoom)
plt.figure(0)
plt.plot(dd_rpp_zoom)

plt.figure(1)
plt.plot(rpp)

plt.figure(2)
plt.plot(tab_srm)
plt.plot(dd_tb_srm)

plt.show()






