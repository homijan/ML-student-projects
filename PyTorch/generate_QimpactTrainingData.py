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
    path = f'gd-profiles/{filename}'
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

# changing units um->cm
x_Qloc/=1e4
x_Qimpact/=1e4
x_Qsnb/=1e4

def getsub(f, x, xref):
    f_cs = CubicSpline(x, f, extrapolate=True)
    evalf = f_cs(xref)
    # Floor the values
    scale = 1e-4
    evalf[evalf < scale * max(evalf)] = scale * max(evalf)
    return evalf

def get_data(x, Z, T, gradT, Kn, n, Qimp, width, step, T_mean, T_std):  
    rad=0     #finds out how many points fits in (!)radius(!) range
    s=0
    while s<=width/2:
        s=xref[rad]-xref[0]
        rad+=1
    
    numPoints = len(Z[0:2*rad:step]) #int(2*rad/step)+1 # the plus one because of integer evaluation
    if (numPoints % 2 == 0):
        print(f'Number of points per field {numPoints} is not odd.')
        print(f'Change step or width.')
        quit()
    numFields = 6 #5 # if including x and Kn
    Qdata=np.empty((0,numFields*numPoints+2), int) #2 * rad "#of points in interval" * 6 "for each phsy quantity" + 2 "for Q and nln"
    print(f'Row length is {numFields*numPoints+2} (each of x, Z, T, gradT, Kn, n {numPoints} points plus Q and nln)') #2*6{x,Z,T,gradT,Kn,n} + 2{qimpact, nln}
    print(f'Number of rows {len(x)-2*rad+1}')
    
    x_c=np.array([]) #saves coordinate of center to index result array 
    
    for ind, _ in enumerate(x):  #x_min=x[ind], x_max=x[ind+2*rad], x_c=x[ind+rad]
        datapoint=np.array([])          
        if ind+2*rad>=len(x)+1:
            break    
        
        else:
            x_c=np.append(x_c, x[rad+ind])                     #will be used as index column for Qdata
            # TODO: what is the appropriate scaling here? Global (max(x)-min(x)) might be to large!
            datapoint=np.append(datapoint, (x[ind:ind+2*rad:step] - x[rad+ind]) / (max(x) - min(x))) #append all point distances in xmin-xmax
            datapoint=np.append(datapoint, Z[ind:ind+2*rad:step]) #append all Zbar in xmin-xmax
            datapoint=np.append(datapoint, T[ind:ind+2*rad:step]) #append all Te in xmin-xmax
            datapoint=np.append(datapoint, gradT[ind:ind+2*rad:step]) #append all gradTe in xmin-xmax
            datapoint=np.append(datapoint, Kn[ind:ind+2*rad:step]) #append all Knudsen number in xmin-xmax
            datapoint=np.append(datapoint, n[ind:ind+2*rad:step]) #append all gradTe in xmin-xmax
            datapoint=np.append(datapoint, Qimp[ind+rad]) #append Qimpact in x_c
            # Find the self-similar nonlinearity of temperature profile given by
            # T = c * (1 - b * x^2)^(1/nln),
            # where c stands for Tc, b for 1/xf^2 and nln = n in Zeldovich
            # Evaluate effective heat flux (logistic weighting of Qloc and Qfs) 
            def Tselfsimilar(X, nln):
                #x, x0, T0, x1, T1 = X
                x = X
                b = (1.0 - (T1 / T0)**nln) / (x1**2.0 - x0**2.0 * (T1 / T0)**nln)
                c = T0 / (1.0 - b * x0**2.0)**(1.0 / nln)
                print(f'T0 {T0} T0(b, c) {c * (1.0 - b * x0**2.0)**(1.0/nln)}')
                print(f'T1 {T1} T1(b, c) {c * (1.0 - b * x1**2.0)**(1.0/nln)}')
                return c * (1.0 - b * x**2.0)**(1.0 / nln)
            xpoints = x[ind:ind+2*rad:step]
            Tpoints = T[ind:ind+2*rad:step].values * T_std + T_mean
            x0 = xpoints[0]; T0 = Tpoints[0]; x1 = xpoints[len(xpoints)-1]; T1 = Tpoints[len(Tpoints)-1]
            par, cov = curve_fit(Tselfsimilar, xpoints, Tpoints, maxfev = 1000)
            plt.plot(xpoints, Tpoints, 'k-')
            plt.plot(xpoints, Tselfsimilar(xpoints, par[0]))
            plt.show()
            #standev=np.sqrt(np.diag(cov))
            print(f'xc {x[rad+ind]}, par {par}, Tavg {0.5*(T0 + T1)}')
            datapoint=np.append(datapoint, max(0.0, min(par[0], 5./2.))) #append nonlinearity of T in x_c bounded by (0,5/2)     
            # Add datapoint
            Qdata=np.append(Qdata,[datapoint], axis=0)
            
            if ind%500==0:
                print(f"We're done with {ind}/{len(x)-2*rad+1} points") 
    
    #Naming of columns 
    column_names=[]
    for _,name in enumerate(['x', 'Z', 'T', 'gradT', 'Kn', 'n']):
        for i in range(numPoints):
            column_names=np.append(column_names,f'{name}_{i}')
    column_names=np.append(column_names,['Q', 'nln'])
    
    df_Qdata=pd.DataFrame(Qdata, columns=column_names, index=x_c)
    print('Done')
    return df_Qdata, numPoints

# Transform data to match given spatial interval 
# Default values spanning the whole c7b spatial domain and number of points
xmin = -1.0; xmax = 1.0; Npoints = 10000; step = 32
# Default kernel size is 100 microns
width = 100e-4
if (len(sys.argv) > 1):
    width = float(sys.argv[1]) * 1e-4
if (len(sys.argv) > 2):
    xmin = float(sys.argv[2])
if (len(sys.argv) > 3):
    xmax = float(sys.argv[3])
if (len(sys.argv) > 4):
    Npoints = int(sys.argv[4])
if (len(sys.argv) > 5):
    step = int(sys.argv[5])
print(f'width={width:.1e}, xmin={xmin:.1e}, xmax={xmax:.1e}, Npoints={Npoints}')

xref = np.linspace(xmin, xmax, Npoints)
Te = getsub(Te, x_Te, xref)
ne = getsub(ne, x_ne, xref)
Zbar = getsub(Zbar, x_Zbar, xref)
Qloc = getsub(Qloc, x_Qloc, xref)
Qimpact = getsub(Qimpact, x_Qimpact, xref)
Qsnb = getsub(Qsnb, x_Qsnb, xref)
Qc7bBGK = getsub(Qc7bBGK, x_Qc7bBGK, xref)
absKnx = -Knx
absKnx = getsub(absKnx, x_Qc7bBGK, xref)
Qc7bAWBS = getsub(Qc7bAWBS, x_Qc7bAWBS, xref)

#calculating Te gradient
gradTe=np.gradient(Te, xref)

path = './'
# Scale the input data
data_scaling=pd.DataFrame(index=['mean', 'std'], columns=['Z', 'T', 'gradT', 'Kn', 'n', 'Q']) #will contain mean values and std
scaled_data=pd.DataFrame([Zbar, Te, gradTe, absKnx, ne, Qimpact], index=['Z', 'T', 'gradT', 'Kn', 'n', 'Q']).T 
# ^^^ All data of which I want to find mean and std 
for val in data_scaling.columns:
    data_scaling[val]['mean']=scaled_data[val].mean()
    data_scaling[val]['std']=scaled_data[val].std()
for col in scaled_data.columns:
    scaled_data[col]=(scaled_data[col]-data_scaling[col].loc['mean'])/data_scaling[col].loc['std']
data_scaling.to_csv(f'{path}/data_scaling.csv')

#Qdata=get_data(xref, Zbar, Te, gradTe, absKnx, ne, Qimpact, width, step)
#Qdata.to_csv(f'{path}/Qdata{step}width{width:.0e}cm.csv')
scaled_Qdata, numPoints=get_data(xref, scaled_data['Z'],  scaled_data['T'],  
                                 scaled_data['gradT'],  scaled_data['Kn'],
                                 scaled_data['n'],  scaled_data['Q'], width, step, 
                                 data_scaling['T']['mean'], data_scaling['T']['std'])
scaled_Qdata.to_csv(f'{path}/scaled_QdataKn{numPoints}width{width*1e4:.0f}microns.csv')
