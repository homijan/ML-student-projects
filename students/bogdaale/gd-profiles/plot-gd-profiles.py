import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

x_Te, Te = np.loadtxt('Te_gdhohlraum_cm_10ps_TekeV_interp.txt', usecols=(0, 1), unpack=True)
x_ne, ne = np.loadtxt('ne_gdhohlraum_cm_ne1e20cm3_interp.txt', usecols=(0, 1), unpack=True)
x_Zbar, Zbar = np.loadtxt('Zbar_gdhohlraum_cm_Z_interp.txt', usecols=(0, 1), unpack=True)

x_Qloc, Qloc = np.loadtxt('Q_gdhohlraum_microns_10ps_LocalWcm2.txt', usecols=(0, 1), unpack=True)
x_Qimpact, Qimpact = np.loadtxt('Q_gdhohlraum_microns_10ps_IMPACTWcm2.txt', usecols=(0, 1), unpack=True)
x_Qsnb, Qsnb = np.loadtxt('Q_gdhohlraum_microns_10ps_separatedsnbWcm2.txt', usecols=(0, 1), unpack=True)
x_Qc7bBGK, Qc7bBGK = np.loadtxt('Q_gdhohlraum_cm_10ps_c7b-bgk-Wcm2-clogCHIC.txt', comments='#', delimiter=', ', usecols=(0, 8), unpack=True)
x_Qc7bAWBS, Qc7bAWBS = np.loadtxt('Q_gdhohlraum_cm_10ps_c7b-awbs-Wcm2-clogCHIC.txt', comments='#', delimiter=', ', usecols=(0, 8), unpack=True)


fontsize = 20#15.5
plt.rcParams.update({'font.size': fontsize})

fig1, axs1 = plt.subplots(1, 1, figsize=(8, 8))
axs1.plot(x_Te, Te, label="Te")
axs1.set_xlabel('cm')
axs1.set_ylabel('eV')
axs1.set_title('Te')
#axs1.autoscale(enable=True, axis='x', tight=True)
axs1.legend()

fig2, axs2 = plt.subplots(1, 1, figsize=(8, 8))
axs2.plot(x_ne, ne, label="ne")
axs2.set_xlabel('cm')
axs2.set_ylabel('1/cm$-3$')
axs2.set_title('ne')
#axs2.autoscale(enable=True, axis='x', tight=True)
axs2.legend()

fig3, axs3 = plt.subplots(1, 1, figsize=(8, 8))
axs3.plot(x_Zbar, Zbar, label="Zbar")
axs3.set_xlabel('cm')
axs3.set_title('Zbar')
#axs3.autoscale(enable=True, axis='x', tight=True)
axs3.legend()

fig4, axs4 = plt.subplots(1, 1, figsize=(8, 8))
axs4.plot(x_Qloc, Qloc, label="Qloc")
axs4.plot(x_Qimpact, Qimpact, label="Qimpact")
axs4.plot(x_Qsnb, Qsnb, label="Qsnb")
#axs4.plot(1e4 * x_Qc7bAWBS, Qc7bAWBS, label="Qc7b-awbs")
axs4.plot(1e4 * x_Qc7bAWBS, gaussian_filter1d(Qc7bAWBS,3), label="Qc7b-awbs")
#axs4.plot(1e4 * x_Qc7bBGK, Qc7bBGK, label="Qc7b-bgk")
axs4.plot(1e4 * x_Qc7bBGK, gaussian_filter1d(Qc7bBGK, 3), label="Qc7b-bgk")
axs4.set_xlabel('$\mu$m')
axs4.set_ylabel('W/cm$^2$')
axs4.set_title('Q')
#axs4.autoscale(enable=True, axis='x', tight=True)
axs4.legend(loc='upper left')

plt.show()
