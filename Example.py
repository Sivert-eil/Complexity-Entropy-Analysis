import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import random

from Complexity_Entropy import ComplexityEntropy, MaxMin_complexity

'''
The file contains examples where the Complexity-Entropy analysis has been 
applied and is for demonstration purposes. 
'''

'''
Code to generate the maximum and minimum complexity lines with the embedding 
dimension chosen for most of the analysis done, and is one of the more common 
embedding dimensions used for the Complexity-Entropy analysis
'''
#Embedding dimension to be used for the rest of the calculations
d = 6


region_CH = MaxMin_complexity(d, n_steps = 1)
Max = region_CH.Maximum() 
region_CH = MaxMin_complexity(d, n_steps = 100)
Min = region_CH.Minimum() 
#%%


plt.figure(figsize = (9,6))
plt.title('Max/Min Complexity lines for Embedding Dimension d = {}'.format(d))
plt.plot(Max[0], Max[1], color = 'tab:blue', label = 'd = {}'.format(d))
plt.plot(Min[0], Min[1], color = 'tab:blue')
plt.legend()
plt.show
##############################################################################
'''
Code to reproduce the maximum and minimum complexity lines for different 
embedding dimensions ass seen in the Theory pdf. 
However, to generate the maximum complexity line for embedding dimensions of 
d = 8 and above takes a long time as the number of points for the lines grows 
as [d! x (d! – 1)]
'''
region_CH_1= MaxMin_complexity(d = 5, n_steps=1)
Max1 = region_CH_1.Maximum()
region_CH_1= MaxMin_complexity(d = 5, n_steps=100)
Min1 = region_CH_1.Minimum()

region_CH_2= MaxMin_complexity(d = 5, n_steps=1)
Max2 = region_CH_2.Maximum()
region_CH_2= MaxMin_complexity(d = 5, n_steps=100)
Min2 = region_CH_2.Minimum()

region_CH_3= MaxMin_complexity(d = 6, n_steps=1)
Max3 = region_CH_3.Maximum()
region_CH_3= MaxMin_complexity(d = 6, n_steps=100)
Min3 = region_CH_3.Minimum()

region_CH_4= MaxMin_complexity(d = 7, n_steps=1)
Max4 = region_CH_4.Maximum()
region_CH_4= MaxMin_complexity(d = 7, n_steps=100)
Min4 = region_CH_4.Minimum()


plt.figure(figsize=(9,6))
plt.title('Maximum and minimum complexity line for different embedding dimensions')
plt.xlabel('Entropy')
plt.ylabel('Complexity')
plt.plot(Max1[0], Max1[1], color = 'tab:red', label = 'd = 4')
plt.plot(Min1[0], Min1[1], color = 'tab:red')

plt.plot(Max2[0], Max2[1], color = 'tab:orange', label = 'd = 5')
plt.plot(Min2[0], Min2[1], color = 'tab:orange')

plt.plot(Max3[0], Max3[1], color = 'tab:blue', label = 'd = 6')
plt.plot(Min3[0], Min3[1], color = 'tab:blue')

plt.plot(Max4[0], Max4[1], color = 'tab:green', label = 'd = 7')
plt.plot(Min4[0], Min4[1], color = 'tab:green')
plt.legend
plt.show()
##############################################################################
#%%
'''
Example code for the Complexity-Entropy analysis of a white noise process
'''
n = 20000

series = [random.gauss(0,1) for i in range(n)]
series = np.array(series)

# Plotting the time series
plt.figure(figsize = (10,4))
plt.title('White noise')
plt.plot(series[:1000], label = 'White noise')
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()

# CH-plane location
CH_whitenoise = ComplexityEntropy(series, d)
H_white, C_white = CH_whitenoise.CH_plane()

plt.figure(figsize=(9,6))
plt.title('Complexity-Entropy analysis White noise')
plt.xlabel('Entropy')
plt.ylabel('Complexity')
plt.plot(Max[0], Max[1], color = 'tab:blue')
plt.plot(Min[0], Min[1], color = 'tab:blue')
plt.plot(H_white, C_white, 'rs', markersize = 8)
plt.show()

##############################################################################
'''
Example code for a linear model. 
'''
n = 10000
time_series = [1 + 0.01*x for x in range(n + 1)]

#Plotting time series
plt.figure(figsize = (10,4))
plt.title(r'Linear Function $f(x) = ax + b$')
plt.plot(time_series)
plt.ylabel(r'$f(x)$', rotation = 'horizontal')
plt.xlabel(r'$x$')
plt.tight_layout()
plt.show()

# CH-plane location
CH_linear = ComplexityEntropy(time_series, d)
H_lin, C_lin = CH_linear.CH_plane()

plt.figure(figsize = (9,6))
plt.title('Complexity-Entropy Linear Function and White noise')
plt.xlabel('Entropy')
plt.ylabel('Complexity')
plt.plot(Max[0], Max[1], color = 'tab:blue')
plt.plot(Min[0], Min[1], color = 'tab:blue')
plt.plot(H_lin, C_lin, 'rs', markersize = 8)
plt.show()

##############################################################################
#%%
'''
Example code for a sine function
'''
# angular frequency
omega =  np.pi

# discretization time
dt = 0.001

N = 10
time_array = np.linspace(0, N, int(N/dt))
time_series_sine = np.sin(omega * time_array)

#Plotting the time series
plt.figure(figsize = (10,4))
plt.title(r'$\sin(\omega t)$')
plt.plot(time_array, time_series_sine)
plt.xlabel(r'$t$')
plt.ylabel(r'$\sin(\omega t)$')
plt.tight_layout()
plt.show()

# CH-plane location
CH_sine = ComplexityEntropy(time_series_sine, d)
H_sine, C_sine = CH_sine.CH_plane()

plt.figure(figsize = (9,6))
plt.title('Complexity-Entropy ' + r'$\sin(\omega t)$')
plt.plot(Max[0], Max[1], color = 'tab:blue')
plt.plot(Min[0], Min[1], color = 'tab:blue')
plt.plot(H_sine, C_sine, 'rs', label = 'Sine', markersize = 8)
plt.xlabel('Entropy')
plt.ylabel('Complexity')
plt.show()

##############################################################################
#%%
'''
Continuous-time models can be resampled to emulate simulating the model with a 
rougher discretization timestep. Result of these analysis shows that the 
continuous time models should be considered with lines and curves in the 
Complexity-Entropy plane rather than with distinct points

This requires a long base time series to obtain accurate results

Resampling Complexity-Entropy analysis applied to sine function
For this you need to make sure the final resampled time series still uphold 
the length requirement for the Complexity-Entropy analysis.
'''

Entropy_sine = list()
Complex_sine = list()

lags = [1 + n for n in range(2000 + 1)]


for x in tqdm(range(len(lags)), desc='Sine function', ncols=70):
    # List comprehensions expression, list[start:end:skip], redefines list 
    #object from "start" to "end" with "skip" being the number of list elements 
    #skipped when resampling the original list
    red_ts = time_series_sine[::lags[x]]
    CH_sine_red = ComplexityEntropy(red_ts, d)
    H_sine_r, C_sine_r = CH_sine_red.CH_plane()
    
    Entropy_sine.append(H_sine_r)
    Complex_sine.append(C_sine_r)
    
plt.figure(figsize = (9,6))

plt.title('Complexity-Entropy ' + r'$\sin(\omega t)$')
plt.plot(Max[0], Max[1], color = 'tab:blue')
plt.plot(Min[0], Min[1], color = 'tab:blue')
plt.plot(Entropy_sine[1:], Complex_sine[1:], 'rs', label = 'Resampled', markersize = 6)
plt.plot(Entropy_sine[0], Complex_sine[0], 'bs', label = 'Original', markersize = 8)
plt.xlabel('Entropy')
plt.ylabel('Complexity')
plt.legend()
plt.show()

##############################################################################
#%%
'''
Logistic Map
'''

# Defining funtion for the logistic map
def logistic_map(N, init_value, growth):
    '''
    Logistic Map:
    x[n+1] = r * x[n] * (1 - x[n])
    
    
    Parameters
    ----------
    N : INT
        Length of the time series
    init_value : FLOAT
        Initial value of the logistic map
    growth : FLOAT
        Growth rate of the logistic map

    Returns
    -------
    X : LIST
        List contianing the entires of the logistic map time series
    '''
    X = [init_value]
    for n in range(N-1):
        x_current = X[n]
        X.append(x_current * growth * (1 - x_current))
    return X

r1 = 3.5
r2 = 3.7
series_1 = logistic_map(10000, 0.5, r1)
series_2 = logistic_map(10000, 0.5, r2)

# Plotting the time series from ordered section of growth parameter, r1
plt.figure(figsize = (10,4))
plt.plot(series_1[0:100], 'o--', label = 'Logistic map, r = {}'.format(r1))
plt.title('Logistic Map, r = {}'.format(r1))
plt.xlabel(r'$n$')
plt.show()


# Plotting the time series from chaotic section of growth parameter, r2
plt.figure(figsize = (10,4))
plt.plot(series_2[0:100], 'o--', label = 'Logistic map, r = {}'.format(r2))
plt.title('Logistic Map, r = {}'.format(r2))
plt.xlabel(r'$n$')
plt.show()

'''
Complexity-Entropy analysis of a scan of the growth parameter of the logistic
map, from r = 3.5 to r = 4.0 with 1000 steps
'''

# Defining lists for entropy/complexity lists
logistic_H = []
logistic_C = []


x = 0.4
for r in tqdm(list(np.linspace(3.5, 4, 1000)), desc='Logistic', ncols=100):
    if r > 4:
        r = 4
    time_s = list(logistic_map(2**15, x, r))
    Logistic_CH = ComplexityEntropy(time_s, d)
    H, C = Logistic_CH.CH_plane()
    logistic_H.append(H)
    logistic_C.append(C)
        
plt.figure(figsize = (9,6))
plt.plot(Max[0], Max[1], color = 'tab:blue')
plt.plot(Min[0], Min[1], color = 'tab:blue')
plt.title('Complexity-Entropy Logistic Map')
plt.xlabel('Entropy')
plt.ylabel('Complexity')
plt.scatter(logistic_H, logistic_C, color = 'blue', label = 'logistic map', s = 20)
plt.show()


##############################################################################
#%% 
''' 
Lorenz model simulation 
code sourced from the Wikipedia article of the Lorenz model
https://en.wikipedia.org/wiki/Lorenz_system
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

#state0 = [1.0, 1.0, 1.0]
state0 = [0, 1.0, 0]
t = np.arange(0.0, 50.0, 0.001) # (start, stop, d_step)

states = odeint(f, state0, t)


fig = plt.figure(figsize=(6,6))
ax = fig.gca(projection="3d")
ax.plot(states[:, 0], states[:, 1], states[:, 2])
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.draw()
plt.tight_layout()
plt.close()


''' Lorenz model Complexity-Entropy analysis '''

Lorenz_CH1 = ComplexityEntropy(states[:, 0][::1], d, tau = 1)
Lorenz_CH2 = ComplexityEntropy(states[:, 1][::1], d, tau = 1)
Lorenz_CH3 = ComplexityEntropy(states[:, 2][::1], d, tau = 1)

H1, C1 = Lorenz_CH1.CH_plane()
H2, C2 = Lorenz_CH2.CH_plane()
H3, C3 = Lorenz_CH3.CH_plane()

plt.figure(figsize = (9,6))
plt.title('Complexity-Entropy Lorenz Model')
plt.plot(Max[0], Max[1], color = 'tab:blue')
plt.plot(Min[0], Min[1], color = 'tab:blue')
plt.plot(H1, C1, 'rs', markersize = 8, label = 'Lorenz Model, x')
plt.plot(H2, C2, 'bs', markersize = 8, label = 'Lorenz Model, y')
plt.plot(H3, C3, 'gs', markersize = 8, label = 'Lorenz Model, z')
plt.xlabel('Entropy')
plt.ylabel('Complexity')
plt.legend()
plt.show()


'''
Same resampling technique used on the sine function, used on the Lorenz model. 
For this you need to make sure the final resampled time series still uphold 
the length requirement for the Complexity-Entropy analysis. 
'''
H_lorenz = []
C_lorenz = []
H2_lorenz = []
C2_lorenz = []
H3_lorenz = []
C3_lorenz = []
delay = [1 + i for i in range(5000)]

for i in tqdm(delay, desc='Lorenz Model', ncols = 70):
    Lorenz_CH = ComplexityEntropy(states[:, 0][::i], d)
    H, C = Lorenz_CH.CH_plane()
    H_lorenz.append(H)
    C_lorenz.append(C)


for i in tqdm(delay, desc='Lorenz Model', ncols = 70):
    Lorenz_CH = ComplexityEntropy(states[:, 1][::i], d)
    H, C = Lorenz_CH.CH_plane()
    H2_lorenz.append(H)
    C2_lorenz.append(C)


for i in tqdm(delay, desc='Lorenz Model', ncols = 70):
    Lorenz_CH = ComplexityEntropy(states[:, 2][::i], d)
    H, C = Lorenz_CH.CH_plane()
    H3_lorenz.append(H)
    C3_lorenz.append(C)


plt.figure(figsize = (9,6))
plt.title('Complexity-Entropy Lorenz Model')
plt.plot(Max[0], Max[1], color = 'tab:blue')
plt.plot(Min[0], Min[1], color = 'tab:blue')
plt.plot(H_lorenz, C_lorenz, 'rs', markersize = 8, label = 'Lorenz Model, x')
plt.plot(H2_lorenz, C2_lorenz, 'bs', markersize = 8, label = 'Lorenz Model, y')
plt.plot(H3_lorenz, C3_lorenz, 'gs', markersize = 8, label = 'Lorenz Model, z')
plt.xlabel('Entropy')
plt.ylabel('Complexity')
plt.legend()
plt.show()


##############################################################################
#%% FBM
''' Fractional Brownian Motion: '''

# Importing the python package for fractional Brownian motion
from fbm import FBM

Hurst = 0.1
# Simulating Fractional Brownian motion
#          (length, hurst exponent)
fBm1 = FBM(500, Hurst) 
series_1 = list(fBm1.fbm())

hurst2 = 0.7
fBm2= FBM(500, hurst2) # length, hurst exponent
series_2 = list(fBm2.fbm())

# Plotting the time series
plt.figure(figsize = (10,4))
plt.plot(series_1, label = 'fBm, h = {}'.format(Hurst))
plt.title('Fractional Brownian motion, h = {}'.format(Hurst))
plt.tight_layout()
plt.show()

# Plotting the time series
plt.figure(figsize = (10,4))
plt.plot(series_2, label = 'fBm, h = {}'.format(hurst2))
plt.title('Fractional Brownian motion, h = {}'.format(hurst2))
plt.tight_layout()
plt.show()


''' Fractional Gaussian Noise '''
series_3 = list(fBm1.fgn())
series_4 = list(fBm2.fgn())

# Plotting the time series
plt.figure(figsize = (10,4))
plt.plot(series_3, label = 'fGn, h = {}'.format(Hurst))
plt.title('Fractional Gaussian noise, h = {}'.format(Hurst))
plt.tight_layout()
plt.show()

# Plotting the time series
plt.figure(figsize = (10,4))
plt.plot(series_4, label = 'fGn, h = {}'.format(0.7))
plt.title('Fractional Gaussian noise, h = {}'.format(0.7))
plt.tight_layout()
plt.show()


'''
Fractional Brownian Motion Complexity-Entropy analysis, should create a nice 
curve in the Complexity-Entropy plane
'''
from fbm import FBM
# Number of realizations to average over
m = 100

# Creating multi-dimensional list to calculate complexity/entropy for several realizations of FBM
Complex_fbm = [[] for i in range(m)]
Entropy_fbm = [[] for i in range(m)]

# Length of the time series
n = 2**19


# List different husrt exponents
hurst = [0.001 , 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 
         0.65, 0.7, 0.75, 0.8, 0.84, 0.88, 0.90, 0.905, 0.91, 0.915, 0.92]

for x in range(m):
    for h in tqdm(hurst, desc = 'FBM {}'.format(x), ncols = 70):
        #Calculate time series frational Brownian motion
        fBm = FBM(n, h)

        ts = list(fBm.fbm())
        FBM_CH = ComplexityEntropy(ts, d)   # Initialize CH class
        H, C = FBM_CH.CH_plane()            # Calculate entropy/complexity
        Complex_fbm[x].append(C)            # Adding values of entropy complexity
        Entropy_fbm[x].append(H)            # to the multi-dimensional list



# Calculating the column mean to get the average complexity/entropy for all realizations of FBM
Complex_fbm = np.mean(Complex_fbm, axis = 0)
Entropy_fbm = np.mean(Entropy_fbm, axis = 0)


plt.figure(figsize = (9,6))
plt.title('Compelxity-Entropy: FBM')
plt.plot(Max[0], Max[1], color = 'tab:blue')
plt.plot(Min[0], Min[1], color = 'tab:blue')
plt.plot(Entropy_fbm, Complex_fbm, 'rs-', markersize = 5)
plt.xlabel('Entropy')
plt.ylabel('Complexity')

# Marking select Hurst exponents in the CH-plane
plt.text(0.96, -0.01, '0.001', size=10, horizontalalignment = 'center')
plt.text(0.93, 0.075, '0.3', size=10, horizontalalignment = 'center')
plt.text(0.87, 0.166, '0.5', size=10, horizontalalignment = 'center')
plt.text(0.69, 0.29, '0.8', size=10, horizontalalignment = 'center')
plt.show()


'''
Fractional Gaussian Noise Complexity-Entropy analysis
'''
# Importing the python package for fractional Brownian motion
from fbm import FBM
# Number of realizations to average over
m = 100

# Creating multi-dimensional list to calculate complexity/entropy for several realizations of FBM
Complex_fgn = [[] for i in range(m)]
Entropy_fgn = [[] for i in range(m)]

# Length of the time series
n = 2**19

# List different husrt exponents
hurst = [0.001 , 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 
         0.65, 0.7, 0.75, 0.8, 0.84, 0.88, 0.90, 0.905, 0.91, 0.915, 0.92]


for x in range(m):
    for h in tqdm(hurst, desc = 'FGN {}'.format(x), ncols = 70):
        #Calculate time series frational Brownian motion
        fBm = FBM(n, h)

        ts = list(fBm.fgn())
        FBM_CH = ComplexityEntropy(ts, d)   # Initialize CH class
        H, C = FBM_CH.CH_plane()            # Calculate entropy/complexity
        Complex_fgn[x].append(C)            # Adding values of entropy complexity
        Entropy_fgn[x].append(H)            # to the multi-dimensional list



# Calculating the column mean to get the average complexity/entropy for all realizations of FBM
Complex_fgn = np.mean(Complex_fgn, axis = 0)
Entropy_fgn = np.mean(Entropy_fgn, axis = 0)


fig = plt.figure(figsize=(12,6))
fig.suptitle('Complexity-Entropy Fractional Gaussian Noise', fontsize=15)
gs = gridspec.GridSpec(1, 2, width_ratios=[2,1])
ax1 = plt.subplot(gs[0])
ax1.set_xlabel('Entropy', fontsize=13)
ax1.set_ylabel('Complexity', fontsize=13)
ax1.plot(Max[0], Max[1], color = 'tab:blue')
ax1.plot(Min[0], Min[1], color = 'tab:blue')
ax1.plot(Entropy_fbm, Complex_fbm, color = 'red', label = 'FBM')


ax1.plot(Entropy_fgn, Complex_fgn, 'bs-', markersize=9, label = 'FGN')

ax1.legend(fontsize=13) #loc='upper left' fontsize=10

ax2 = plt.subplot(gs[1])
ax2.plot(Max[0], Max[1], color = 'tab:blue')
ax2.plot(Min[0], Min[1], color = 'tab:blue')
ax2.plot(Entropy_fbm, Complex_fbm, color = 'red', label = 'FBM')

ax2.plot(Entropy_fgn[:10], Complex_fgn[:10], 'bs-', markersize=9, label = 'FGN')
ax2.set_xlim(0.9, 1.005)
ax2.set_ylim(-0.01, 0.1)
ax2.set_xlabel('Entropy', fontsize=13)
ax2.set_ylabel('Complexity', fontsize=13)
fig.tight_layout()

# Marking select Hurst exponents in the CH-plane
plt.text(0.98, 0.08, '0.001', size=13, horizontalalignment = 'center')
plt.text(0.99, -0.001, '0.5', size=13, horizontalalignment = 'center')
plt.show()