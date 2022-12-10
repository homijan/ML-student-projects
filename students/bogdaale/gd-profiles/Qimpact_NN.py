import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy.interpolate import CubicSpline
import math
import sys

#!!!
def impdata(filename):
    #path=f'https://raw.githubusercontent.com/homijan/ML-student-projects/intro-ab/students/bogdaale/gd-profiles/{filename}'
    path = f'./{filename}'
    return path
#!!!

x_Te, Te = np.loadtxt(impdata('Te_gdhohlraum_cm_10ps_TekeV_interp.txt'), usecols=(0, 1), unpack=True)
x_ne, ne = np.loadtxt(impdata('ne_gdhohlraum_cm_ne1e20cm3_interp.txt'), usecols=(0, 1), unpack=True)
x_Zbar, Zbar = np.loadtxt(impdata('Zbar_gdhohlraum_cm_Z_interp.txt'), usecols=(0, 1), unpack=True)

x_Qloc, Qloc = np.loadtxt(impdata('Q_gdhohlraum_microns_10ps_LocalWcm2.txt'), usecols=(0, 1), unpack=True)
x_Qimpact, Qimpact = np.loadtxt(impdata('Q_gdhohlraum_microns_10ps_IMPACTWcm2.txt'), usecols=(0, 1), unpack=True)
x_Qsnb, Qsnb = np.loadtxt(impdata('Q_gdhohlraum_microns_10ps_separatedsnbWcm2.txt'), usecols=(0, 1), unpack=True)

x_Qc7bBGK, Qc7bBGK, Knx = np.loadtxt(impdata('Q_gdhohlraum_cm_10ps_c7b-bgk-Wcm2-clogCHIC.txt'), comments='#', delimiter=', ', usecols=(0, 8, 6), unpack=True)
x_Qc7bAWBS, Qc7bAWBS = np.loadtxt(impdata('Q_gdhohlraum_cm_10ps_c7b-awbs-Wcm2-clogCHIC.txt'), comments='#', delimiter=', ', usecols=(0, 8), unpack=True)

x_QNN, QNN = np.loadtxt(impdata('Qimpact-NN.txt'), usecols=(0, 1), unpack=True)



# changing units um->cm
x_Qloc/=1e4
x_Qimpact/=1e4
x_Qsnb/=1e4

def getsub(f, x, xref):
  f_cs = CubicSpline(x, f)
  return f_cs(xref)

def Qstream(ne, Te):
  me = 9.1094e-28 # [g]
  eV2K = 1.1604e4 # K = eV2K * eV
  erg2J = 1e-7
  kB = 1.3807e-16 * eV2K # [erg/eV]
  # Local thermal energy density
  eTh = ne * kB * Te
  # Thermal velocity
  vTh = (kB * Te / me)**0.5
  # Free-streaming heat flux [cm/s*erg/cm3]
  Qfs = vTh * eTh
  return erg2J * Qfs

# Transform data to match given spatial interval 
# Default values spanning the whole c7b spatial domain
xmin = -1.0; xmax = 1.0
if (len(sys.argv) > 1):
  xmin = float(sys.argv[1])
if (len(sys.argv) > 2):
  xmax = float(sys.argv[2])

xref = x_Te[np.logical_and(x_Te > xmin, x_Te < xmax)]
Te = getsub(Te, x_Te, xref)
ne = getsub(ne, x_ne, xref)
Zbar = getsub(Zbar, x_Zbar, xref)
Qloc = getsub(Qloc, x_Qloc, xref)
Qimpact = getsub(Qimpact, x_Qimpact, xref)
Qsnb = getsub(Qsnb, x_Qsnb, xref)
QNN = getsub(QNN, x_QNN, xref)
Qc7bBGK = getsub(Qc7bBGK, x_Qc7bBGK, xref)
Knx = getsub(Knx, x_Qc7bBGK, xref)
Qc7bAWBS = getsub(Qc7bAWBS, x_Qc7bAWBS, xref)

#calculating Te gradient
gradTe=np.gradient(Te, xref)

# Evaluate free-streaming heat flux
Qfs = Qstream(ne, Te)

# Evaluate effective heat flux (logistic weighting of Qloc and Qfs) 
def fitQeff(X, flim):
    #fit function for Qloc profile
    ne, Z, Te, gradTe = X
    kQSH = 6.1e+02 # scaling constant corresponding to the SHICK local heat flux
    Qloc = -(kQSH/Z)*((Z+0.24)/(Z+4.2))*Te**2.5*gradTe
    Qfs = Qstream(ne, Te)
    Qeff = flim * Qfs * (1.0 - np.exp(-Qloc/(flim*Qfs)))
    return Qeff
par3, cov3 = curve_fit(fitQeff, (ne, Zbar, Te, gradTe), Qimpact,  maxfev = 1000)
standev3=np.sqrt(np.diag(cov3))
flim = par3[0]
print(f'Flux limiter from Qeff profile flim = {flim:.1e} ± {standev3[0]:.1e}')

if (len(sys.argv) > 3):
  flim = float(sys.argv[3])
  print(f'Flux limiter overruled to flim = {flim}')
    
#plot stuff
fontsize = 16
plt.rcParams.update({'font.size': fontsize})

strflim = f'{flim:.2f}'
fig1, axs1 = plt.subplots(1, 1, figsize=(12, 8))
axs1.plot(xref, Qimpact, 'm', label="Impact (kinetic reference)", linewidth=4.0)
axs1.plot(xref, QNN, 'c--', linewidth=3.0, label=r'Q with variable $f(x)$ by NN')
axs1.plot(xref, Qsnb, 'g-.', label="Q by Schurtz-Nikolai-Busquet")
axs1.plot(xref, 0.1*Qloc, 'k-.', label=r'Local Q$_{SH}$ x 0.1')
axs1.plot(xref, fitQeff((ne, Zbar, Te, gradTe), flim), 'g-', label=f'Q limited at const $f$={strflim}')
#axs1.plot(xref, flim*Qfs, ':', label=f'{strflim} free-streaming')
axs1.plot(xref, Te/max(Te)*0.1*max(Qloc), 'r:', label=f'Te in ({min(Te):.0f}, {max(Te):.0f}) eV')
axs1.plot(xref, ne/max(ne)*0.1*max(Qloc), 'b:', label=f'ne in ({min(ne):.1e}, {max(ne):.1e}) cm-3')
axs1.set_xlabel('cm')
axs1.set_ylabel('W/cm$^2$')
axs1.legend(loc="upper left")
axs1.autoscale(enable=True, axis='x', tight=True)
axs1.set_title(r'NN-driven spatial varying limiter $f(x)$')

fig2, axs2 = plt.subplots(1, 1, figsize=(12, 8))
axs2.plot(xref, Zbar, label="Zbar")
axs2.set_xlabel('cm')
axs2.set_title('Zbar')
axs2.autoscale(enable=True, axis='x', tight=True)
axs2.legend()

if (False):
  fig3, axs3 = plt.subplots(1, 1, figsize=(12, 8))
  axs3.plot(xref, Qloc, label="Qloc")
  axs3.plot(xref, Qimpact, label="Qimpact")
  axs3.plot(xref, Qsnb, label="Qsnb")
  #axs3.plot(xref, Qc7bAWBS, label="Qc7b-awbs")
  axs3.plot(xref, gaussian_filter1d(Qc7bAWBS,3), label="Qc7b-awbs")
  #axs3.plot(xref, Qc7bBGK, label="Qc7b-bgk")
  axs3.plot(xref, gaussian_filter1d(Qc7bBGK, 3), label="Qc7b-bgk")
  axs3.set_xlabel('cm')
  axs3.set_ylabel('W/cm$^2$')
  axs3.set_title('Q')
  #axs3.autoscale(enable=True, axis='x', tight=True)
  axs3.legend(loc='upper left')

fig4, axs4 = plt.subplots(1, 1, figsize=(12, 8))
axs4.plot(xref, Knx)
axs4.set_xlabel('cm')
axs4.set_ylabel('[-]')
axs4.set_title(label=r"Knudsen number $Kn_{\mathrm{x}}$")
axs4.autoscale(enable=True, axis='x', tight=True)

plt.show()
