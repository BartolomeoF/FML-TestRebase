###################
# Loading modules #
###################

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

##################
# Plotting style #
##################

plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 28
#plt.rc('xtick.major', pad=20)
#plt.rc('ytick.major', pad=15)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.labelsize'] = 22#32
plt.rcParams['ytick.labelsize'] = 22#32
plt.rcParams['axes.labelpad'] = 15
plt.rcParams["errorbar.capsize"] = 8
plt.rcParams["lines.markersize"] = 6 #4, 8
plt.rcParams["lines.markeredgewidth"] = 1
#plt.rcParams["axes.markeredgecolor"] = 'black'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.minor.width'] = 1.5

CB_color_cycle = ['#377eb8', # 0 blue
                  '#ff7f00', # 1 orange
                  '#4daf4a', # 2 green
                  '#f781bf', # 3 pink
                  '#a65628', # 4 brown
                  '#984ea3', # 5 purple
                  '#999999', # 6 grey
                  '#e41a1c', # 7 red
                  '#dede00'  # 8 yellow
                 ]


#####################
# Generic functions #
#####################

# reading
def read_data(input_filename_as_string):
    pofk_data = np.loadtxt(input_filename_as_string)
    #print(np.shape(pofk_data))
    k_arr = pofk_data[:,0]
    P_arr = pofk_data[:,1]
    return k_arr, P_arr

def read_data3(input_filename_as_string):
    data = np.loadtxt(input_filename_as_string)
    #print(np.shape(data))
    a_arr = data[:,0]
    chioverdelta_arr = data[:,1]
    coupl_arr = data[:,2]
    return a_arr, chioverdelta_arr, coupl_arr

def read_data_powmes(input_filename_as_string, box, Npart):
    pofk_data = np.loadtxt(input_filename_as_string)
    wave_num_arr = pofk_data[1:,0]
    P_rough_arr = pofk_data[1:,3]
    shot_noise_arr = pofk_data[1:,4]
    k_arr = [wv*2.*np.pi/box for wv in wave_num_arr]
    P_arr = [box**3.*(Prv-shotv/(Npart**3.)) for Prv, shotv in zip(P_rough_arr, shot_noise_arr)]
    return k_arr, P_arr

def read_data_powmes_five(input_filename_as_string1, input_filename_as_string2, input_filename_as_string3, input_filename_as_string4, input_filename_as_string5, box, Npart):
    k_arr1, P_arr1 = read_data_powmes(input_filename_as_string1, box, Npart)
    k_arr2, P_arr2 = read_data_powmes(input_filename_as_string2, box, Npart)
    k_arr3, P_arr3 = read_data_powmes(input_filename_as_string3, box, Npart)
    k_arr4, P_arr4 = read_data_powmes(input_filename_as_string4, box, Npart)
    k_arr5, P_arr5 = read_data_powmes(input_filename_as_string5, box, Npart)
    P_avg_arr = [(P1+P2+P3+P4+P5)/5. for P1, P2, P3, P4, P5 in zip(P_arr1, P_arr2, P_arr3, P_arr4, P_arr5)]
    return k_arr1, P_avg_arr

def read_data_Barreira_input(input_filename_as_string):
    pofk_data = np.loadtxt(input_filename_as_string)
    k_arr = pofk_data[1:,0]
    P_arr = pofk_data[1:,1]
    return k_arr, P_arr

def read_data_FML_cosmo(input_filename_as_string):
    pofk_data = np.loadtxt(input_filename_as_string)
    #print(np.shape(pofk_data))
    a_arr = pofk_data[1:,0]
    E_arr = pofk_data[1:,1]
    E_prime_E_arr = pofk_data[1:,2]
    return a_arr, E_arr, E_prime_E_arr

def read_data_FML_growth(input_filename_as_string):
    pofk_data = np.loadtxt(input_filename_as_string)
    #print(np.shape(pofk_data))
    a_arr = pofk_data[1:,0]
    GeffOverG_arr = pofk_data[1:,1]
    D1_arr = pofk_data[1:,2]
    return a_arr, GeffOverG_arr, D1_arr

# writing to file
def write_data(a_arr_inv, E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr, output_filename_as_string):
    datafile_id = open(output_filename_as_string, 'wb')    #here you open the ascii file
    data = np.array([np.array(a_arr_inv[::-1]), np.array(E_arr[::-1]), np.array(alpha1_arr[::-1]), np.array(alpha2_arr[::-1]), np.array(B_arr[::-1]), np.array(C_arr[::-1])])
    data = data.T     #here you transpose your data, so to have it in two columns
    np.savetxt(datafile_id, data, fmt=['%.4e','%.4e','%.4e','%.4e','%.4e','%.4e'])    #here the ascii file is populated.
    datafile_id.close()    #close the file

def write_data_coupl(a_arr_inv, E_arr, B_arr, C_arr, Coupl_arr, output_filename_as_string):
    datafile_id = open(output_filename_as_string, 'wb')    #here you open the ascii file
    data = np.array([np.array(a_arr_inv[::-1]), np.array(E_arr[::-1]), np.array(B_arr[::-1]), np.array(C_arr[::-1]), np.array(Coupl_arr[::-1])])
    data = data.T     #here you transpose your data, so to have it in two columns
    np.savetxt(datafile_id, data, fmt=['%.4e','%.4e','%.4e','%.4e','%.4e'])    #here the ascii file is populated.
    datafile_id.close()    #close the file

def write_data_screencoupl(a_arr_inv, chioverdelta_arr, Coupl_arr, output_filename_as_string):
    datafile_id = open(output_filename_as_string, 'wb')    #here you open the ascii file
    data = np.array([np.array(a_arr_inv[::-1]), np.array(chioverdelta_arr[::-1]), np.array(Coupl_arr[::-1])])
    data = data.T     #here you transpose your data, so to have it in two columns
    np.savetxt(datafile_id, data, fmt=['%.4e','%.4e','%.4e'])    #here the ascii file is populated.
    datafile_id.close()    #close the file

def comp_diff(k_arr1, P_arr1, k_arr2, P_arr2):
    #print(k_arr1[0], k_arr1[-1], k_arr2[0], k_arr2[-1])
    P_interp = interp1d(k_arr1, P_arr1, fill_value="extrapolate")
    P_interp_arr = P_interp(k_arr2)
    P_ratio = [x/y for x, y in zip(P_interp_arr, P_arr2)] #[x/y-1. for x, y in zip(P_interp_arr, P_arr2)]
    return k_arr2, P_ratio

#take ratio
def take_ratio1(k_arr1, P_arr1, k_arr2, P_arr2):
    #print(k_arr1[0], k_arr1[-1], k_arr2[0], k_arr2[-1])
    P_interp = interp1d(k_arr1, P_arr1, fill_value="extrapolate")
    P_interp_arr = P_interp(k_arr2)
    P_ratio = [x/y for x, y in zip(P_interp_arr, P_arr2)]
    return k_arr2, P_ratio

def take_ratio2(k_arr1, P_arr1, k_arr2, P_arr2):
    #print(k_arr1[0], k_arr1[-1], k_arr2[0], k_arr2[-1])
    P_interp = interp1d(k_arr2, P_arr2, fill_value="extrapolate")
    P_interp_arr = P_interp(k_arr1)
    P_ratio = [x/y for x, y in zip(P_arr1, P_interp_arr)]
    return k_arr1, P_ratio

##########################
# Cosmological functions #
##########################

def comp_Omega_r_LCDM(z, Omega_r0, Omega_m0):
    Omega_L0 = 1.-Omega_m0-Omega_r0
    E = comp_E_LCDM(z, Omega_r0, Omega_m0)
    Omega_r = Omega_r0*(1.+z)**4./E/E
    return Omega_r

def comp_Omega_m_LCDM(z, Omega_r0, Omega_m0):
    Omega_L0 = 1.-Omega_m0-Omega_r0
    E = comp_E_LCDM(z, Omega_r0, Omega_m0)
    Omega_m = Omega_m0*(1.+z)**3./E/E
    return Omega_m

def comp_Omega_L_LCDM(z, Omega_r0, Omega_m0):
    Omega_L0 = 1.-Omega_m0-Omega_r0
    E = comp_E_LCDM(z, Omega_r0, Omega_m0)
    Omega_L = Omega_L0/E/E
    return Omega_L

def comp_E_LCDM(z, Omega_r0, Omega_m0):
    Omega_L0 = 1.-Omega_m0-Omega_r0
    E = np.sqrt(Omega_m0*(1.+z)**3. + Omega_r0*(1.+z)**4. + Omega_L0)
    return E

def comp_E_prime_E_LCDM(z, Omega_r0, Omega_m0):
    Omega_DE0 = 1. - Omega_m0 - Omega_r0
    term1 = Omega_r0*(1.+z)**4.+Omega_m0*(1.+z)**3.+Omega_DE0
    term2 = 4.*Omega_r0*(1.+z)**4.+3.*Omega_m0*(1.+z)**3.
    E_prime_E = -0.5*term2/term1
    return E_prime_E

def comp_w_eff(E_prime_E):
    w_eff = -1. - 2.*E_prime_E/3.
    return w_eff

def comp_Omega_r_prime(Omega_r, E, E_prime):
    E_prime_E = E_prime/E
    Omega_r_prime = -Omega_r*(4.+2.*E_prime_E)
    return Omega_r_prime

def comp_Omega_m_prime(Omega_m, E, E_prime):
    E_prime_E = E_prime/E
    Omega_m_prime = -Omega_m*(3.+2.*E_prime_E)
    return Omega_m_prime

def comp_Omega_L_prime(Omega_L, E, E_prime):
    E_prime_E = E_prime/E
    Omega_L_prime = -2.*Omega_L*E_prime_E
    return Omega_L_prime

def comp_B(alpha0, alpha1, alpha2, beta0):
    if beta0 == 0:
        B = 0
    else:
        B = 4.*beta0/(alpha0+2.*alpha1*alpha2+alpha2*alpha2)
    return B

def comp_C(alpha0, alpha1, alpha2):
    if alpha1 == 0 and alpha2 == 0:
        C = 0
    else:
        C = (alpha1+alpha2)/(alpha0+2.*alpha1*alpha2+alpha2*alpha2)
    return C

def comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0):
    chioverdelta_arr = [Bv*Cv*Omega_m0/av/av/av/Ev/Ev for av, Ev, Bv, Cv in zip(a_arr_inv, E_arr, B_arr, C_arr)]
    return chioverdelta_arr

def comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr):
    Coupl_arr = [-1.*(alpha_1v+alpha_2v)*Cv for alpha_1v, alpha_2v, Cv in zip(alpha1_arr, alpha2_arr, C_arr)]
    return Coupl_arr

def comp_screen_fac_Lombriser(chioverdelta, dens):
    chi = dens*chioverdelta
    screen_fac = 2.*((1.+chi)**0.5-1.)/chi
    return screen_fac

def comp_screen_fac_Winther(a, chioverdelta, dens):
    chi = (1.+dens)*chioverdelta*a*a*a
    screen_fac = 2.*((1.+chi)**0.5-1.)/chi
    return screen_fac

#####################################
# Cubic Galileon dS limit functions #
#####################################

# 1st Friedman eq.
def fried0_cuGaldS(y, Omr, Omm, OmL, E, E_dS):
    alpha_fac = 1.-OmL/E_dS/E_dS
    u = E/E_dS
    term1 = alpha_fac*y**2.*(2.*y*u**2.-1.)
    zer = 1.-Omr-Omm-OmL-term1
    return zer

# find the root of the 1st Friedman eq. to yield y
def comp_y_cuGal_dS(y_guess, Omr, Omm, OmL, E, E_dS):
    y = fsolve(fried0_cuGaldS, y_guess, args=(Omr, Omm, OmL, E, E_dS))[0]
    return y

def comp_E_prime_E_cuGal_dS(Omega_r, Omega_L, y, E, E_dS, alpha_fac):
    u = E/E_dS
    numer1 = 3.*alpha_fac*y**2.
    numer2 = -6.*alpha_fac*u**2.*y**3.*(1.-u**2.*y)/(1.-2.*u**2.*y)
    numer3 = -Omega_r+3.*Omega_L-3.
    numer = numer1+numer2+numer3
    denom = 2.*(1.+alpha_fac*u**4.*y**4./(2.*u**2.*y-1.))
    E_prime_E = numer/denom
    return E_prime_E

def comp_E_prime_E_cuGal_dS_numer(Omega_r, Omega_L, y, E, E_dS, alpha_fac):
    u = E/E_dS
    numer1 = 3.*alpha_fac*y**2.
    numer2 = -6.*alpha_fac*u**2.*y**3.*(1.-u**2.*y)/(1.-2.*u**2.*y)
    numer3 = -Omega_r+3.*Omega_L-3.
    numer = numer1+numer2+numer3
    denom = 2.*(1.+alpha_fac*u**4.*y**4./(2.*u**2.*y-1.))
    E_prime_E = numer/denom
    return numer

def comp_E_prime_E_cuGal_dS_numer1(Omega_r, Omega_L, y, E, E_dS, alpha_fac):
    u = E/E_dS
    numer1 = 3.*alpha_fac*y**2.
    numer2 = -6.*alpha_fac*u**2.*y**3.*(1.-u**2.*y)/(1.-2.*u**2.*y)
    numer3 = -Omega_r+3.*Omega_L-3.
    numer = numer1+numer2+numer3
    denom = 2.*(1.+alpha_fac*u**4.*y**4./(2.*u**2.*y-1.))
    E_prime_E = numer/denom
    return numer1

def comp_E_prime_E_cuGal_dS_numer2(Omega_r, Omega_L, y, E, E_dS, alpha_fac):
    u = E/E_dS
    numer1 = 3.*alpha_fac*y**2.
    numer2 = -6.*alpha_fac*u**2.*y**3.*(1.-u**2.*y)/(1.-2.*u**2.*y)
    numer3 = -Omega_r+3.*Omega_L-3.
    numer = numer1+numer2+numer3
    denom = 2.*(1.+alpha_fac*u**4.*y**4./(2.*u**2.*y-1.))
    E_prime_E = numer/denom
    return numer2

def comp_E_prime_E_cuGal_dS_numer3(Omega_r, Omega_L, y, E, E_dS, alpha_fac):
    u = E/E_dS
    numer1 = 3.*alpha_fac*y**2.
    numer2 = -6.*alpha_fac*u**2.*y**3.*(1.-u**2.*y)/(1.-2.*u**2.*y)
    numer3 = -Omega_r+3.*Omega_L-3.
    numer = numer1+numer2+numer3
    denom = 2.*(1.+alpha_fac*u**4.*y**4./(2.*u**2.*y-1.))
    E_prime_E = numer/denom
    return numer3

def comp_E_prime_E_cuGal_dS_denom(Omega_r, Omega_L, y, E, E_dS, alpha_fac):
    u = E/E_dS
    numer1 = 3.*alpha_fac*y**2.
    numer2 = -6.*alpha_fac*u**2.*y**3.*(1.-u**2.*y)/(1.-2.*u**2.*y)
    numer3 = -Omega_r+3.*Omega_L-3.
    numer = numer1+numer2+numer3
    denom = 2.*(1.+alpha_fac*u**4.*y**4./(2.*u**2.*y-1.))
    E_prime_E = numer/denom
    return 2.*u**2.*y-1.#denom

def comp_y_prime_cuGal_dS(y, E, E_dS, E_prime_E):
    u = E/E_dS
    term1 = 3.*y/(2.*y*u**2.-1.)
    term2 = -y**2.*u**2.*(E_prime_E+3.)/(2.*y*u**2.-1.)
    term3 = -E_prime_E*y
    y_prime = term1 + term2 + term3
    return y_prime

def comp_Omphi_Fried_cuGal_dS(y, E, E_dS, alpha_fac):
    Omega_phi = alpha_fac*y**2.*(2.*y*E**2./E_dS/E_dS - 1.)
    return Omega_phi

def comp_OmDE_prime_cuGal_dS(y, y_prime, E, E_prime, E_dS):
    term1 = 6.*y**2.*y_prime*E**2./E_dS/E_dS
    term2 = 4.*y**3.*E_prime*E/E_dS/E_dS
    term3 = -2.*y*y_prime
    OmDE_prime = term1+term2+term3
    return OmDE_prime

def comp_alpha0_cuGal_dS(y, y_prime, E, E_prime, E_dS, alpha_fac):
    term1 = 3.*alpha_fac*y*y
    term2 = -2.*alpha_fac*E*E*y*y*y*(y_prime/y+E_prime/E+2.)/E_dS/E_dS
    alpha0 = term1 + term2
    return alpha0

def comp_alpha1_cuGal_dS():
    alpha1 = 0.
    return alpha1

def comp_alpha2_cuGal_dS(y, E, E_dS, alpha_fac):
    alpha2 = alpha_fac*E*E*y*y*y/E_dS/E_dS
    return alpha2

def comp_beta0_cuGal_dS(y, E, E_dS, alpha_fac):
    beta0 = alpha_fac*E*E*y*y*y/E_dS/E_dS
    return beta0

# describes the system of ODEs
def comp_primes_cuGal_dS(Y, x, E_dS, alpha_fac):
    y, E, Omega_r, Omega_m, Omega_L = Y
    u = E/E_dS
    E_prime_E = comp_E_prime_E_cuGal_dS(Omega_r, Omega_L, y, E, E_dS, alpha_fac)
    E_prime = E_prime_E*E
    y_prime = comp_y_prime_cuGal_dS(y, E, E_dS, E_prime_E)
    Omega_r_prime = comp_Omega_r_prime(Omega_r, E, E_prime)
    Omega_m_prime = comp_Omega_m_prime(Omega_m, E, E_prime)
    Omega_L_prime = comp_Omega_L_prime(Omega_L, E, E_prime)
    Y_prime = [y_prime, E_prime, Omega_r_prime, Omega_m_prime, Omega_L_prime]
    return Y_prime

def solve_inv_cuGal_dS(z_num, z_ini, Omega_r0, Omega_m0, E_dS_fac, f):
    E0 = comp_E_LCDM(0., Omega_r0, Omega_m0)
    E_dS = E0*E_dS_fac #0.93 #*1e-30
    Omega_L0 = (1.-f)*(1.-Omega_r0-Omega_m0)
    Omega_phi0 = f*(1.-Omega_r0-Omega_m0) #1.-Omega_r0-Omega_m0-Omega_L0
    alpha_fac = 1.-Omega_L0/E_dS/E_dS
    #if alpha_fac<0:
    #    print('Warning, negative alpha')
    if f==0:
        y0 = 0
    else:
        y0 = comp_y_cuGal_dS(0.9, Omega_r0, Omega_m0, Omega_L0, E0, E_dS)
    track = (E0/E_dS)**2.*y0-1.
    #if track < 0:
    #    print('Warning, (E0/E_dS)**2.*y0-1.<0:', track)
    #    return 0

    z_final = 0.
    x_ini = np.log(1./(1.+z_ini))
    x_final = np.log(1./(1.+z_final))
    x_arr = np.linspace(x_ini, x_final, z_num)
    a_arr = [np.exp(x) for x in x_arr]
    z_arr = [1./a - 1. for a in a_arr]
    x_arr_inv = x_arr[::-1]
    a_arr_inv = a_arr[::-1]
    z_arr_inv = z_arr[::-1]

    Y0 = [y0, E0, Omega_r0, Omega_m0, Omega_L0]
    Ep_E0 = comp_E_prime_E_cuGal_dS(Omega_r0, Omega_L0, y0, E0, E_dS, alpha_fac)
    ans = odeint(comp_primes_cuGal_dS, Y0, x_arr_inv, args=(E_dS,alpha_fac))
    y_arr = ans[:,0]
    E_arr = ans[:,1]
    Omega_r_arr = ans[:,2]
    Omega_m_arr = ans[:,3]
    Omega_L_arr = ans[:,4]
    return a_arr_inv, y_arr, E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, E_dS, alpha_fac

# solves the ODEs
def run_solver_inv_cuGal_dS(z_num, z_ini, Omega_r0, Omega_m0, E_dS_fac, f):
    a_arr_inv, y_arr, E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, E_dS, alpha_fac = solve_inv_cuGal_dS(z_num, z_ini, Omega_r0, Omega_m0, E_dS_fac, f)

    Omega_phi_arr = [1.-Omrv-Ommv-OmLv for Omrv, Ommv, OmLv in zip(Omega_r_arr, Omega_m_arr, Omega_L_arr)]
    Omega_phi_arr2  = [comp_Omphi_Fried_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev in zip(y_arr, E_arr)]
    track_arr = [(Ev/E_dS)**2.*yv-1. for Ev, yv in zip(E_arr, y_arr)]
    E_prime_E_arr = [comp_E_prime_E_cuGal_dS(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_arr = [Ep_Ev*Ev for Ep_Ev, Ev in zip(E_prime_E_arr, E_arr)]
    y_prime_arr = [comp_y_prime_cuGal_dS(yv, Ev, E_dS, E_prime_Ev) for yv, Ev, E_prime_Ev in zip(y_arr, E_arr, E_prime_E_arr)]
    Omega_r_prime_arr = [comp_Omega_r_prime(Omrv, Ev, Epv) for Omrv, Ev, Epv in zip(Omega_r_arr, E_arr, E_prime_arr)]
    Omega_m_prime_arr = [comp_Omega_m_prime(Ommv, Ev, Epv) for Ommv, Ev, Epv in zip(Omega_m_arr, E_arr, E_prime_arr)]
    Omega_L_prime_arr = [comp_Omega_L_prime(OmLv, Ev, Epv) for OmLv, Ev, Epv in zip(Omega_L_arr, E_arr, E_prime_arr)]
    #Omega_DE_prime_arr = [comp_OmDE_prime_cuGal_dS(yv, y_primev, Ev, E_primev, E_dS) for yv, y_primev, Ev, E_primev in zip(y_arr, y_prime_arr, E_arr, E_prime_arr)]
    alpha0_arr = [comp_alpha0_cuGal_dS(yv, y_primev, Ev, E_primev, E_dS, alpha_fac) for yv, y_primev, Ev, E_primev in zip(y_arr, y_prime_arr, E_arr, E_prime_arr)]
    alpha1_arr = [comp_alpha1_cuGal_dS() for yv, Ev, in zip(y_arr, E_arr)]
    alpha2_arr = [comp_alpha2_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev, in zip(y_arr, E_arr)]
    beta0_arr = [comp_beta0_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev, in zip(y_arr, E_arr)]
    B_arr = [comp_B(alpha0v, alpha1v, alpha2v, beta0v) for alpha0v, alpha1v, alpha2v, beta0v in zip(alpha0_arr, alpha1_arr, alpha2_arr, beta0_arr)]
    C_arr = [comp_C(alpha0v, alpha1v, alpha2v) for alpha0v, alpha1v, alpha2v in zip(alpha0_arr, alpha1_arr, alpha2_arr)]
    return a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr

def run_solver_inv_cuGal_dS_bg(z_num, z_ini, Omega_r0, Omega_m0, E_dS_fac, f):
    a_arr_inv, y_arr, E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, E_dS, alpha_fac = solve_inv_cuGal_dS(z_num, z_ini, Omega_r0, Omega_m0, E_dS_fac, f)

    Omega_phi_arr = [1.-Omrv-Ommv-OmLv for Omrv, Ommv, OmLv in zip(Omega_r_arr, Omega_m_arr, Omega_L_arr)]
    Omega_phi_arr2  = [comp_Omphi_Fried_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev in zip(y_arr, E_arr)]
    track_arr = [(Ev/E_dS)**2.*yv-1. for Ev, yv in zip(E_arr, y_arr)]
    E_prime_E_arr = [comp_E_prime_E_cuGal_dS(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_arr = [Ep_Ev*Ev for Ep_Ev, Ev in zip(E_prime_E_arr, E_arr)]
    y_prime_arr = [comp_y_prime_cuGal_dS(yv, Ev, E_dS, E_prime_Ev) for yv, Ev, E_prime_Ev in zip(y_arr, E_arr, E_prime_E_arr)]
    Omega_r_prime_arr = [comp_Omega_r_prime(Omrv, Ev, Epv) for Omrv, Ev, Epv in zip(Omega_r_arr, E_arr, E_prime_arr)]
    Omega_m_prime_arr = [comp_Omega_m_prime(Ommv, Ev, Epv) for Ommv, Ev, Epv in zip(Omega_m_arr, E_arr, E_prime_arr)]
    Omega_L_prime_arr = [comp_Omega_L_prime(OmLv, Ev, Epv) for OmLv, Ev, Epv in zip(Omega_L_arr, E_arr, E_prime_arr)]
    #Omega_DE_prime_arr = [comp_OmDE_prime_cuGal_dS(yv, y_primev, Ev, E_primev, E_dS) for yv, y_primev, Ev, E_primev in zip(y_arr, y_prime_arr, E_arr, E_prime_arr)]
    alpha0_arr = [comp_alpha0_cuGal_dS(yv, y_primev, Ev, E_primev, E_dS, alpha_fac) for yv, y_primev, Ev, E_primev in zip(y_arr, y_prime_arr, E_arr, E_prime_arr)]
    alpha1_arr = [comp_alpha1_cuGal_dS() for yv, Ev, in zip(y_arr, E_arr)]
    alpha2_arr = [comp_alpha2_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev, in zip(y_arr, E_arr)]
    beta0_arr = [comp_beta0_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev, in zip(y_arr, E_arr)]
    B_arr = [comp_B(alpha0v, alpha1v, alpha2v, beta0v) for alpha0v, alpha1v, alpha2v, beta0v in zip(alpha0_arr, alpha1_arr, alpha2_arr, beta0_arr)]
    C_arr = [comp_C(alpha0v, alpha1v, alpha2v) for alpha0v, alpha1v, alpha2v in zip(alpha0_arr, alpha1_arr, alpha2_arr)]
    #print(a_arr_inv, E_arr, E_prime_E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, Omega_phi_arr, Omega_phi_arr2, y_arr)
    return a_arr_inv, E_arr, E_prime_E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, Omega_phi_arr, Omega_phi_arr2, y_arr

def run_solver_inv_cuGal_dS_bg2(z_num, z_ini, Omega_r0, Omega_m0, E_dS_fac, f):
    a_arr_inv, y_arr, E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, E_dS, alpha_fac = solve_inv_cuGal_dS(z_num, z_ini, Omega_r0, Omega_m0, E_dS_fac, f)

    Omega_phi_arr = [1.-Omrv-Ommv-OmLv for Omrv, Ommv, OmLv in zip(Omega_r_arr, Omega_m_arr, Omega_L_arr)]
    Omega_phi_arr2  = [comp_Omphi_Fried_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev in zip(y_arr, E_arr)]
    track_arr = [(Ev/E_dS)**2.*yv-1. for Ev, yv in zip(E_arr, y_arr)]
    E_prime_E_arr = [comp_E_prime_E_cuGal_dS(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_E_numer1_arr = [comp_E_prime_E_cuGal_dS_numer1(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_E_numer2_arr = [comp_E_prime_E_cuGal_dS_numer2(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_E_numer3_arr = [comp_E_prime_E_cuGal_dS_numer3(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_E_denom_arr = [comp_E_prime_E_cuGal_dS_denom(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_arr = [Ep_Ev*Ev for Ep_Ev, Ev in zip(E_prime_E_arr, E_arr)]
    y_prime_arr = [comp_y_prime_cuGal_dS(yv, Ev, E_dS, E_prime_Ev) for yv, Ev, E_prime_Ev in zip(y_arr, E_arr, E_prime_E_arr)]
    Omega_r_prime_arr = [comp_Omega_r_prime(Omrv, Ev, Epv) for Omrv, Ev, Epv in zip(Omega_r_arr, E_arr, E_prime_arr)]
    Omega_m_prime_arr = [comp_Omega_m_prime(Ommv, Ev, Epv) for Ommv, Ev, Epv in zip(Omega_m_arr, E_arr, E_prime_arr)]
    Omega_L_prime_arr = [comp_Omega_L_prime(OmLv, Ev, Epv) for OmLv, Ev, Epv in zip(Omega_L_arr, E_arr, E_prime_arr)]
    #Omega_DE_prime_arr = [comp_OmDE_prime_cuGal_dS(yv, y_primev, Ev, E_primev, E_dS) for yv, y_primev, Ev, E_primev in zip(y_arr, y_prime_arr, E_arr, E_prime_arr)]
    alpha0_arr = [comp_alpha0_cuGal_dS(yv, y_primev, Ev, E_primev, E_dS, alpha_fac) for yv, y_primev, Ev, E_primev in zip(y_arr, y_prime_arr, E_arr, E_prime_arr)]
    alpha1_arr = [comp_alpha1_cuGal_dS() for yv, Ev, in zip(y_arr, E_arr)]
    alpha2_arr = [comp_alpha2_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev, in zip(y_arr, E_arr)]
    beta0_arr = [comp_beta0_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev, in zip(y_arr, E_arr)]
    B_arr = [comp_B(alpha0v, alpha1v, alpha2v, beta0v) for alpha0v, alpha1v, alpha2v, beta0v in zip(alpha0_arr, alpha1_arr, alpha2_arr, beta0_arr)]
    C_arr = [comp_C(alpha0v, alpha1v, alpha2v) for alpha0v, alpha1v, alpha2v in zip(alpha0_arr, alpha1_arr, alpha2_arr)]
    #print(a_arr_inv, E_arr, E_prime_E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, Omega_phi_arr, Omega_phi_arr2, y_arr)
    return a_arr_inv, E_prime_E_arr, E_prime_E_numer1_arr, E_prime_E_numer2_arr, E_prime_E_numer3_arr, E_prime_E_denom_arr

def run_solver_inv_cuGal_dS_bg3(z_num, z_ini, Omega_r0, Omega_m0, E_dS_fac, f):
    a_arr_inv, y_arr, E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, E_dS, alpha_fac = solve_inv_cuGal_dS(z_num, z_ini, Omega_r0, Omega_m0, E_dS_fac, f)

    Omega_phi_arr = [1.-Omrv-Ommv-OmLv for Omrv, Ommv, OmLv in zip(Omega_r_arr, Omega_m_arr, Omega_L_arr)]
    Omega_phi_arr2  = [comp_Omphi_Fried_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev in zip(y_arr, E_arr)]
    track_arr = [(Ev/E_dS)**2.*yv-1. for Ev, yv in zip(E_arr, y_arr)]
    E_prime_E_arr = [comp_E_prime_E_cuGal_dS(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_E_numer_arr = [comp_E_prime_E_cuGal_dS_numer(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_E_numer1_arr = [comp_E_prime_E_cuGal_dS_numer1(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_E_numer2_arr = [comp_E_prime_E_cuGal_dS_numer2(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_E_numer3_arr = [comp_E_prime_E_cuGal_dS_numer3(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_E_denom_arr = [comp_E_prime_E_cuGal_dS_denom(Omrv, OmLv, yv, Ev, E_dS, alpha_fac) for yv, Ev, Omrv, OmLv in zip(y_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_arr = [Ep_Ev*Ev for Ep_Ev, Ev in zip(E_prime_E_arr, E_arr)]
    y_prime_arr = [comp_y_prime_cuGal_dS(yv, Ev, E_dS, E_prime_Ev) for yv, Ev, E_prime_Ev in zip(y_arr, E_arr, E_prime_E_arr)]
    Omega_r_prime_arr = [comp_Omega_r_prime(Omrv, Ev, Epv) for Omrv, Ev, Epv in zip(Omega_r_arr, E_arr, E_prime_arr)]
    Omega_m_prime_arr = [comp_Omega_m_prime(Ommv, Ev, Epv) for Ommv, Ev, Epv in zip(Omega_m_arr, E_arr, E_prime_arr)]
    Omega_L_prime_arr = [comp_Omega_L_prime(OmLv, Ev, Epv) for OmLv, Ev, Epv in zip(Omega_L_arr, E_arr, E_prime_arr)]
    #Omega_DE_prime_arr = [comp_OmDE_prime_cuGal_dS(yv, y_primev, Ev, E_primev, E_dS) for yv, y_primev, Ev, E_primev in zip(y_arr, y_prime_arr, E_arr, E_prime_arr)]
    alpha0_arr = [comp_alpha0_cuGal_dS(yv, y_primev, Ev, E_primev, E_dS, alpha_fac) for yv, y_primev, Ev, E_primev in zip(y_arr, y_prime_arr, E_arr, E_prime_arr)]
    alpha1_arr = [comp_alpha1_cuGal_dS() for yv, Ev, in zip(y_arr, E_arr)]
    alpha2_arr = [comp_alpha2_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev, in zip(y_arr, E_arr)]
    beta0_arr = [comp_beta0_cuGal_dS(yv, Ev, E_dS, alpha_fac) for yv, Ev, in zip(y_arr, E_arr)]
    B_arr = [comp_B(alpha0v, alpha1v, alpha2v, beta0v) for alpha0v, alpha1v, alpha2v, beta0v in zip(alpha0_arr, alpha1_arr, alpha2_arr, beta0_arr)]
    C_arr = [comp_C(alpha0v, alpha1v, alpha2v) for alpha0v, alpha1v, alpha2v in zip(alpha0_arr, alpha1_arr, alpha2_arr)]
    #print(a_arr_inv, E_arr, E_prime_E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, Omega_phi_arr, Omega_phi_arr2, y_arr)
    one_arr = [Ev*Ev*yv/E_dS/E_dS-1. for Ev, yv in zip(E_arr, y_arr)]
    two_arr = [2.*Ev*Ev*yv/E_dS/E_dS-1. for Ev, yv in zip(E_arr, y_arr)]
    return a_arr_inv, E_prime_E_arr, E_prime_E_numer_arr, E_prime_E_numer1_arr, E_prime_E_numer2_arr, E_prime_E_numer3_arr, E_prime_E_denom_arr, one_arr, two_arr

def cuGal_dS_viability_test(Omega_r0, Omega_m0, E_dS_fac, f):
    E0 = comp_E_LCDM(0., Omega_r0, Omega_m0)
    E_dS = E0*E_dS_fac #0.93 #*1e-30
    Omega_L0 = (1.-f)*(1.-Omega_r0-Omega_m0)
    Omega_phi0 = f*(1.-Omega_r0-Omega_m0) #1.-Omega_r0-Omega_m0-Omega_L0
    alpha_fac = 1.-Omega_L0/E_dS/E_dS
    #if alpha_fac<0:
        #print('Warning, negative alpha for ', E_dS_fac, ', ', f)
    y0 = comp_y_cuGal_dS(0.9, Omega_r0, Omega_m0, Omega_L0, E0, E_dS)
    track = (E0/E_dS)**2.*y0-1.
    Ep_E0 = comp_E_prime_E_cuGal_dS(Omega_r0, Omega_L0, y0, E0, E_dS, alpha_fac)
    #print(Ep_E0)
    if Ep_E0 > 0:
        #print('Eprime pos', f, E_dS_fac, alpha_fac, y0, track, Ep_E0)
        return alpha_fac, y0, track, Ep_E0
    elif alpha_fac < 0 and y0 > 0:
        #print('alpha neg: ', f, E_dS_fac, alpha_fac, y0, track, Ep_E0)
        return alpha_fac, y0, track, Ep_E0
    elif track < 0:
        #print('Warning, (E0/E_dS)**2.*y0-1.<0:', track)
        #print('(E_dS_fac, f)=(', E_dS_fac, ', ', f, ') is not viable')
        #print('track neg: ', f, E_dS_fac, alpha_fac, y0, track, Ep_E0)
        return alpha_fac, y0, track, Ep_E0
    else:
        #print('(E_dS_fac, f)=(', E_dS_fac, ', ', f, ') is viable')
        return alpha_fac, y0, track, Ep_E0

def comp_almost_track(E_dS_fac, Omega_r0, Omega_m0, f, almost):
    E0 = comp_E_LCDM(0., Omega_r0, Omega_m0)
    E_dS = E0*E_dS_fac #0.93 #*1e-30
    Omega_L0 = (1.-f)*(1.-Omega_r0-Omega_m0)
    y0 = comp_y_cuGal_dS(0.9, Omega_r0, Omega_m0, Omega_L0, E0, E_dS)
    track = (E0/E_dS)**2.*y0-1.
    almost_track = track - almost
    return almost_track

def comp_E_dS_max(E_dS_max_guess, Omr, Omm, f, almost):
    if f == 0:
        if almost < 1e-3:
            f = 1e-3
        else:
            f = almost
    E_dS_max = fsolve(comp_almost_track, E_dS_max_guess, args=(Omr, Omm, f, almost))[0]
    return E_dS_max

# run the code
def runwrite_intermediate_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f, output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f)
    write_data(a_arr_inv, E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr, output_filename_as_string)

def runwrite_coupl_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f, output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    write_data_coupl(a_arr_inv, E_arr, B_arr, C_arr, Coupl_arr, output_filename_as_string)

def run_background_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f)
    return a_arr_inv, E_arr, E_prime_E_arr

def run_screencoupl_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    return a_arr_inv, chioverdelta_arr, Coupl_arr

def run_background_screencoupl_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    return a_arr_inv, E_arr, E_prime_E_arr, chioverdelta_arr, Coupl_arr

def runwrite_background_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f, expansion_output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f)
    write_data_screencoupl(a_arr_inv, E_arr, E_prime_E_arr, expansion_output_filename_as_string)

def runwrite_screencoupl_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f, output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    write_data_screencoupl(a_arr_inv, chioverdelta_arr, Coupl_arr, output_filename_as_string)

def runwrite_background_screencoupl_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f, expansion_output_filename_as_string, screencoupl_output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_dS(z_num, z_max, Omega_r0, Omega_m0, E_dS_fac, f)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    write_data_screencoupl(a_arr_inv, E_arr, E_prime_E_arr, expansion_output_filename_as_string)
    write_data_screencoupl(a_arr_inv, chioverdelta_arr, Coupl_arr, screencoupl_output_filename_as_string)

########################################
# Cubic Galileon today limit functions #
########################################

def comp_E_prime_E_cuGal_today(w, E, Omega_r, Omega_L, k1, g1):
    if w == 0:
        E_prime_E = 0.5*(-Omega_r+3.*Omega_L-3.)
    else:
        numer1 = -0.5*k1*w*w
        numer2 = -3.*g1*E*E*w*w*w*(k1+3.*g1*E*E*w)/(k1+6.*g1*E*E*w)
        numer3 = -Omega_r+3.*Omega_L-3.
        numer = numer1+numer2+numer3
        denom = 2. + 3.*g1*g1*E**4.*w**4./(k1+6.*g1*E*E*w)
        E_prime_E = numer/denom
    return E_prime_E

def comp_E_prime_E_cuGal_today_numer(w, E, Omega_r, Omega_L, k1, g1):
    numer1 = -0.5*k1*w*w
    if w == 0:
        numer2 = 0
    else:
        numer2 = -3.*g1*E*E*w*w*w*(k1+3.*g1*E*E*w)/(k1+6.*g1*E*E*w)
    numer3 = -Omega_r+3.*Omega_L-3.
    numer = numer1+numer2+numer3
    return numer

def comp_E_prime_E_cuGal_today_numer1(w, E, Omega_r, Omega_L, k1, g1):
    numer1 = -0.5*k1*w*w
    return numer1

def comp_E_prime_E_cuGal_today_numer2(w, E, Omega_r, Omega_L, k1, g1):
    if w == 0:
        numer2 = 0
    else:
        numer2 = -3.*g1*E*E*w*w*w*(k1+3.*g1*E*E*w)/(k1+6.*g1*E*E*w)
    return numer2

def comp_E_prime_E_cuGal_today_numer3(w, E, Omega_r, Omega_L, k1, g1):
    numer3 = -Omega_r+3.*Omega_L-3.
    return numer3

def comp_E_prime_E_cuGal_today_denom(w, E, Omega_r, Omega_L, k1, g1):
    if w == 0:
        denom = 2.
    else:
        denom = 2. + 3.*g1*g1*E**4.*w**4./(k1+6.*g1*E*E*w)
    return denom

def comp_w_prime_cuGal_today(w, E, E_prime_E, k1, g1):
    if w == 0:
        w_prime = 0
    else:
        term1 = 3.*k1*w
        term2 = 3.*g1*E*E*w*w*(3.+E_prime_E)
        term3 = k1+6.*g1*E*E*w
        term4 = -E_prime_E*w
        w_prime = -1.*(term1+term2)/term3 + term4
    return w_prime

def comp_Omphi_Fried_cuGal_today(w, E, k1, g1):
    Omega_phi = k1*w*w/6. + g1*E*E*w*w*w
    return Omega_phi

def comp_alpha0_cuGal_today(w, w_prime, E, E_prime, k1, g1):
    if w == 0:
        alpha0 = 0
    else:
        w_prime_w = w_prime/w
        E_prime_E = E_prime/E
        term1 = -0.5*k1*w*w
        term2 = -g1*E*E*w*w*w*(2.+w_prime_w+E_prime_E)
        alpha0 = term1 + term2
    return alpha0

def comp_alpha1_cuGal_today():
    alpha1 = 0.
    return alpha1

def comp_alpha2_cuGal_today(w, E, g1):
    alpha2 = 0.5*g1*E*E*w*w*w
    return alpha2

def comp_beta0_cuGal_today(w, E, g1):
    beta0 = 0.5*g1*E*E*w*w*w
    return beta0

# describes the system of ODEs
def comp_primes_cuGal_today(Y, x, k1, g1):
    a=np.exp(x)
    w, E, Omega_r, Omega_m, Omega_L = Y
    E_prime_E = comp_E_prime_E_cuGal_today(w, E, Omega_r, Omega_L, k1, g1)
    E_prime_E_numer = comp_E_prime_E_cuGal_today_numer(w, E, Omega_r, Omega_L, k1, g1)
    E_prime_E_numer1 = comp_E_prime_E_cuGal_today_numer1(w, E, Omega_r, Omega_L, k1, g1)
    E_prime_E_numer2 = comp_E_prime_E_cuGal_today_numer2(w, E, Omega_r, Omega_L, k1, g1)
    E_prime_E_numer3 = comp_E_prime_E_cuGal_today_numer3(w, E, Omega_r, Omega_L, k1, g1)
    E_prime_E_denom = comp_E_prime_E_cuGal_today_denom(w, E, Omega_r, Omega_L, k1, g1)
    E_prime = E_prime_E*E
    w_prime = comp_w_prime_cuGal_today(w, E, E_prime_E, k1, g1)
    Omega_r_prime = comp_Omega_r_prime(Omega_r, E, E_prime)
    Omega_m_prime = comp_Omega_m_prime(Omega_m, E, E_prime)
    Omega_L_prime = comp_Omega_L_prime(Omega_L, E, E_prime)
    Y_prime = [w_prime, E_prime, Omega_r_prime, Omega_m_prime, Omega_L_prime]
    #print(a, ' Y=', Y)
    #print(a, ' Y_prime=', Y_prime)
    #print(a, E_prime_E, E_prime_E_numer, E_prime_E_numer1, E_prime_E_numer2, E_prime_E_numer3,  E_prime_E_denom, 3.*g1*g1*E**4.*w**4./(k1+6.*g1*E*E*w), 3.*g1*g1*E**4.*w**4., (k1+6.*g1*E*E*w))#k1+3.*g1*E*E*w, k1+6.*g1*E*E*w)
    return Y_prime

def comp_g1(Omega_r0, Omega_m0, k1, f):
    g1 = f*(1.-Omega_r0-Omega_m0)-k1/6.
    return g1

def solve_inv_cuGal_today(z_num, z_ini, Omega_r0, Omega_m0, k1, f):
    if f==0:
        w0=0
    else:
        w0 = 1.
    E0 = 1. #comp_E_LCDM(0., Omega_r0, Omega_m0)
    Omega_L0 = (1.-f)*(1.-Omega_r0-Omega_m0)
    Omega_phi0 = f*(1.-Omega_r0-Omega_m0) #1.-Omega_r0-Omega_m0-Omega_L0
    g1 = comp_g1(Omega_r0, Omega_m0, k1, f)
    #print('k1=', k1, ' g1=', g1)

    z_final = 0.
    x_ini = np.log(1./(1.+z_ini))
    x_final = np.log(1./(1.+z_final))
    x_arr = np.linspace(x_ini, x_final, z_num)
    a_arr = [np.exp(x) for x in x_arr]
    z_arr = [1./a - 1. for a in a_arr]
    x_arr_inv = x_arr[::-1]
    a_arr_inv = a_arr[::-1]
    z_arr_inv = z_arr[::-1]

    Y0 = [w0, E0, Omega_r0, Omega_m0, Omega_L0]
    Ep_E0 = comp_E_prime_E_cuGal_today(w0, E0, Omega_r0, Omega_L0, k1, g1)
    ans = odeint(comp_primes_cuGal_today, Y0, x_arr_inv, args=(k1,g1))#, hmax=1e-3)
    w_arr = ans[:,0]
    E_arr = ans[:,1]
    Omega_r_arr = ans[:,2]
    Omega_m_arr = ans[:,3]
    Omega_L_arr = ans[:,4]
    return a_arr_inv, w_arr, E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, g1

# solves the ODEs
def run_solver_inv_cuGal_today(z_num, z_ini, Omega_r0, Omega_m0, k1, f):
    a_arr_inv, w_arr, E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, g1 = solve_inv_cuGal_today(z_num, z_ini, Omega_r0, Omega_m0, k1, f)

    Omega_phi_arr = [1.-Omrv-Ommv-OmLv for Omrv, Ommv, OmLv in zip(Omega_r_arr, Omega_m_arr, Omega_L_arr)]
    Omega_phi_arr2  = [comp_Omphi_Fried_cuGal_today(wv, Ev, k1, g1) for wv, Ev in zip(w_arr, E_arr)]
    E_prime_E_arr = [comp_E_prime_E_cuGal_today(wv, Ev, Omrv, OmLv, k1, g1) for wv, Ev, Omrv, OmLv in zip(w_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_arr = [Ep_Ev*Ev for Ep_Ev, Ev in zip(E_prime_E_arr, E_arr)]
    w_prime_arr = [comp_w_prime_cuGal_today(wv, Ev, E_prime_Ev, k1, g1) for wv, Ev, E_prime_Ev in zip(w_arr, E_arr, E_prime_E_arr)]
    Omega_r_prime_arr = [comp_Omega_r_prime(Omrv, Ev, Epv) for Omrv, Ev, Epv in zip(Omega_r_arr, E_arr, E_prime_arr)]
    Omega_m_prime_arr = [comp_Omega_m_prime(Ommv, Ev, Epv) for Ommv, Ev, Epv in zip(Omega_m_arr, E_arr, E_prime_arr)]
    Omega_L_prime_arr = [comp_Omega_L_prime(OmLv, Ev, Epv) for OmLv, Ev, Epv in zip(Omega_L_arr, E_arr, E_prime_arr)]
    #Omega_DE_prime_arr = [comp_OmDE_prime_cuGal_today(yv, y_primev, Ev, E_primev, E_dS) for yv, y_primev, Ev, E_primev in zip(y_arr, y_prime_arr, E_arr, E_prime_arr)]
    alpha0_arr = [comp_alpha0_cuGal_today(wv, w_primev, Ev, E_primev, k1, g1) for wv, w_primev, Ev, E_primev in zip(w_arr, w_prime_arr, E_arr, E_prime_arr)]
    alpha1_arr = [comp_alpha1_cuGal_today() for wv, Ev, in zip(w_arr, E_arr)]
    alpha2_arr = [comp_alpha2_cuGal_today(wv, Ev, g1) for wv, Ev, in zip(w_arr, E_arr)]
    beta0_arr = [comp_beta0_cuGal_today(wv, Ev, g1) for wv, Ev, in zip(w_arr, E_arr)]
    B_arr = [comp_B(alpha0v, alpha1v, alpha2v, beta0v) for alpha0v, alpha1v, alpha2v, beta0v in zip(alpha0_arr, alpha1_arr, alpha2_arr, beta0_arr)]
    C_arr = [comp_C(alpha0v, alpha1v, alpha2v) for alpha0v, alpha1v, alpha2v in zip(alpha0_arr, alpha1_arr, alpha2_arr)]
    return a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr

def run_solver_inv_cuGal_today_bg(z_num, z_ini, Omega_r0, Omega_m0, k1, f):
    a_arr_inv, w_arr, E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, g1 = solve_inv_cuGal_today(z_num, z_ini, Omega_r0, Omega_m0, k1, f)

    Omega_phi_arr = [1.-Omrv-Ommv-OmLv for Omrv, Ommv, OmLv in zip(Omega_r_arr, Omega_m_arr, Omega_L_arr)]
    Omega_phi_arr2  = [comp_Omphi_Fried_cuGal_today(wv, Ev, k1, g1) for wv, Ev in zip(w_arr, E_arr)]
    E_prime_E_arr = [comp_E_prime_E_cuGal_today(wv, Ev, Omrv, OmLv, k1, g1) for wv, Ev, Omrv, OmLv in zip(w_arr, E_arr, Omega_r_arr, Omega_L_arr)]
    E_prime_arr = [Ep_Ev*Ev for Ep_Ev, Ev in zip(E_prime_E_arr, E_arr)]
    w_prime_arr = [comp_w_prime_cuGal_today(wv, Ev, E_prime_Ev, k1, g1) for wv, Ev, E_prime_Ev in zip(w_arr, E_arr, E_prime_E_arr)]
    Omega_r_prime_arr = [comp_Omega_r_prime(Omrv, Ev, Epv) for Omrv, Ev, Epv in zip(Omega_r_arr, E_arr, E_prime_arr)]
    Omega_m_prime_arr = [comp_Omega_m_prime(Ommv, Ev, Epv) for Ommv, Ev, Epv in zip(Omega_m_arr, E_arr, E_prime_arr)]
    Omega_L_prime_arr = [comp_Omega_L_prime(OmLv, Ev, Epv) for OmLv, Ev, Epv in zip(Omega_L_arr, E_arr, E_prime_arr)]
    #Omega_DE_prime_arr = [comp_OmDE_prime_cuGal_today(yv, y_primev, Ev, E_primev, E_dS) for yv, y_primev, Ev, E_primev in zip(y_arr, y_prime_arr, E_arr, E_prime_arr)]
    alpha0_arr = [comp_alpha0_cuGal_today(wv, w_primev, Ev, E_primev, k1, g1) for wv, w_primev, Ev, E_primev in zip(w_arr, w_prime_arr, E_arr, E_prime_arr)]
    alpha1_arr = [comp_alpha1_cuGal_today() for wv, Ev, in zip(w_arr, E_arr)]
    alpha2_arr = [comp_alpha2_cuGal_today(wv, Ev, g1) for wv, Ev, in zip(w_arr, E_arr)]
    beta0_arr = [comp_beta0_cuGal_today(wv, Ev, g1) for wv, Ev, in zip(w_arr, E_arr)]
    B_arr = [comp_B(alpha0v, alpha1v, alpha2v, beta0v) for alpha0v, alpha1v, alpha2v, beta0v in zip(alpha0_arr, alpha1_arr, alpha2_arr, beta0_arr)]
    C_arr = [comp_C(alpha0v, alpha1v, alpha2v) for alpha0v, alpha1v, alpha2v in zip(alpha0_arr, alpha1_arr, alpha2_arr)]
    return a_arr_inv, E_arr, E_prime_E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, Omega_phi_arr, Omega_phi_arr2, w_arr

# run the code
def runwrite_intermediate_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f, output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f)
    write_data(a_arr_inv, E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr, output_filename_as_string)

def runwrite_coupl_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f, output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    write_data_coupl(a_arr_inv, E_arr, B_arr, C_arr, Coupl_arr, output_filename_as_string)

def run_background_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f)
    return a_arr_inv, E_arr, E_prime_E_arr

def run_screencoupl_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    return a_arr_inv, chioverdelta_arr, Coupl_arr

def run_background_screencoupl_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    return a_arr_inv, E_arr, E_prime_E_arr, chioverdelta_arr, Coupl_arr

def runwrite_background_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f, expansion_output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f)
    write_data_screencoupl(a_arr_inv, E_arr, E_prime_E_arr, expansion_output_filename_as_string)

def runwrite_screencoupl_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f, output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    write_data_screencoupl(a_arr_inv, chioverdelta_arr, Coupl_arr, output_filename_as_string)

def runwrite_background_screencoupl_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f, expansion_output_filename_as_string, screencoupl_output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_today(z_num, z_max, Omega_r0, Omega_m0, k1, f)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    write_data_screencoupl(a_arr_inv, E_arr, E_prime_E_arr, expansion_output_filename_as_string)
    write_data_screencoupl(a_arr_inv, chioverdelta_arr, Coupl_arr, screencoupl_output_filename_as_string)

#################
# DGP functions #
#################

def comp_dlogHdloga_LCDM(a, Omega_r0, Omega_m0):
    z =1./a-1.
    E = comp_E_LCDM(z, Omega_r0, Omega_m0)
    dlogHdloga = 1./(2.*E*E) * (-3.*Omega_m0/a/a/a - 4.*Omega_r0/a/a/a/a)
    return dlogHdloga

def comp_betaDGP3(a, Omega_m0, Omega_rc0):
    rcH0 = 0.5/np.sqrt(Omega_rc0)
    z=1./a-1.
    E = comp_E_LCDM(z, 0., Omega_m0)
    dlogHdloga = comp_dlogHdloga_LCDM(a, 0., Omega_m0)
    betaDGP = 1. + 2.*rcH0*E*(1.+dlogHdloga/3.)
    return betaDGP

def comp_coupling_DGP(beta_DGP):
    coupl_DGP = 1./3./beta_DGP
    return coupl_DGP

def comp_chioverdelta_DGP(a, beta_DGP, Omega_m0, Omega_rc):
    chioverdelta_DGP = 2.*Omega_m0/(9.*Omega_rc*beta_DGP*beta_DGP*a*a*a)
    return chioverdelta_DGP

def run_background_DGP(z_num, z_ini, Omega_r0_DGP, Omega_m0_DGP, rcH0_DGP):
    Omega_rc0_DGP = 0.25/rcH0_DGP/rcH0_DGP
    z_final = 0.
    x_ini = np.log(1./(1.+z_ini))
    x_final = np.log(1./(1.+z_final))
    x_arr = np.linspace(x_ini, x_final, z_num)
    a_arr = [np.exp(x) for x in x_arr]
    a_arr_inv = a_arr[::-1]
    E_arr = [comp_E_LCDM(1./av-1., Omega_r0_DGP, Omega_m0_DGP) for av in a_arr_inv]
    E_prime_E_arr = [comp_E_prime_E_LCDM(1./av-1., Omega_r0_DGP, Omega_m0_DGP) for av in a_arr_inv]
    return a_arr_inv, E_arr, E_prime_E_arr

def run_screencoupl_DGP(z_num, z_ini, Omega_m0_DGP, rcH0_DGP):
    Omega_rc0_DGP = 0.25/rcH0_DGP/rcH0_DGP
    z_final = 0.
    x_ini = np.log(1./(1.+z_ini))
    x_final = np.log(1./(1.+z_final))
    x_arr = np.linspace(x_ini, x_final, z_num)
    a_arr = [np.exp(x) for x in x_arr]
    a_arr_inv = a_arr[::-1]
    beta_DGP_arr = [comp_betaDGP3(av, Omega_m0_DGP, Omega_rc0_DGP) for av in a_arr_inv]
    chioverdelta_arr = [comp_chioverdelta_DGP(av, beta_DGPv, Omega_m0_DGP, Omega_rc0_DGP) for av, beta_DGPv in zip(a_arr_inv, beta_DGP_arr)]
    Coupl_arr = [comp_coupling_DGP(beta_DGPv) for beta_DGPv in beta_DGP_arr]
    return a_arr_inv, chioverdelta_arr, Coupl_arr

def run_background_screencoupl_DGP(z_num, z_ini, Omega_r0_DGP, Omega_m0_DGP, rcH0_DGP):
    Omega_rc0_DGP = 0.25/rcH0_DGP/rcH0_DGP
    z_final = 0.
    x_ini = np.log(1./(1.+z_ini))
    x_final = np.log(1./(1.+z_final))
    x_arr = np.linspace(x_ini, x_final, z_num)
    a_arr = [np.exp(x) for x in x_arr]
    a_arr_inv = a_arr[::-1]
    E_arr = [comp_E_LCDM(1./av-1., Omega_r0_DGP, Omega_m0_DGP) for av in a_arr_inv]
    E_prime_E_arr = [comp_E_prime_E_LCDM(1./av-1., Omega_r0_DGP, Omega_m0_DGP) for av in a_arr_inv]
    beta_DGP_arr = [comp_betaDGP3(av, Omega_m0_DGP, Omega_rc0_DGP) for av in a_arr_inv]
    chioverdelta_arr = [comp_chioverdelta_DGP(av, beta_DGPv, Omega_m0_DGP, Omega_rc0_DGP) for av, beta_DGPv in zip(a_arr_inv, beta_DGP_arr)]
    Coupl_arr = [comp_coupling_DGP(beta_DGPv) for beta_DGPv in beta_DGP_arr]
    return a_arr_inv, E_arr, E_prime_E_arr, chioverdelta_arr, Coupl_arr

def runwrite_background_DGP(z_num, z_ini, Omega_r0_DGP, Omega_m0_DGP, rcH0_DGP, expansion_output_filename_as_string):
    Omega_rc0_DGP = 0.25/rcH0_DGP/rcH0_DGP
    z_final = 0.
    x_ini = np.log(1./(1.+z_ini))
    x_final = np.log(1./(1.+z_final))
    x_arr = np.linspace(x_ini, x_final, z_num)
    a_arr = [np.exp(x) for x in x_arr]
    a_arr_inv = a_arr[::-1]
    E_arr = [comp_E_LCDM(1./av-1., Omega_r0_DGP, Omega_m0_DGP) for av in a_arr_inv]
    E_prime_E_arr = [comp_E_prime_E_LCDM(1./av-1., Omega_r0_DGP, Omega_m0_DGP) for av in a_arr_inv]
    write_data_screencoupl(a_arr_inv, E_arr, E_prime_E_arr, expansion_output_filename_as_string)

def runwrite_screencoupl_DGP(z_num, z_ini, Omega_m0_DGP, rcH0_DGP, output_filename_as_string):
    Omega_rc0_DGP = 0.25/rcH0_DGP/rcH0_DGP
    z_final = 0.
    x_ini = np.log(1./(1.+z_ini))
    x_final = np.log(1./(1.+z_final))
    x_arr = np.linspace(x_ini, x_final, z_num)
    a_arr = [np.exp(x) for x in x_arr]
    a_arr_inv = a_arr[::-1]
    beta_DGP_arr = [comp_betaDGP3(av, Omega_m0_DGP, Omega_rc0_DGP) for av in a_arr_inv]
    chioverdelta_arr = [comp_chioverdelta_DGP(av, beta_DGPv, Omega_m0_DGP, Omega_rc0_DGP) for av, beta_DGPv in zip(a_arr_inv, beta_DGP_arr)]
    Coupl_arr = [comp_coupling_DGP(beta_DGPv) for beta_DGPv in beta_DGP_arr]
    write_data_screencoupl(a_arr_inv, chioverdelta_arr, Coupl_arr, output_filename_as_string)

def runwrite_background_screencoupl_DGP(z_num, z_ini, Omega_r0_DGP, Omega_m0_DGP, rcH0_DGP, expansion_output_filename_as_string, screencoupl_output_filename_as_string):
    Omega_rc0_DGP = 0.25/rcH0_DGP/rcH0_DGP
    z_final = 0.
    x_ini = np.log(1./(1.+z_ini))
    x_final = np.log(1./(1.+z_final))
    x_arr = np.linspace(x_ini, x_final, z_num)
    a_arr = [np.exp(x) for x in x_arr]
    a_arr_inv = a_arr[::-1]
    E_arr = [comp_E_LCDM(1./av-1., Omega_r0_DGP, Omega_m0_DGP) for av in a_arr_inv]
    E_prime_E_arr = [comp_E_prime_E_LCDM(1./av-1., Omega_r0_DGP, Omega_m0_DGP) for av in a_arr_inv]
    beta_DGP_arr = [comp_betaDGP3(av, Omega_m0_DGP, Omega_rc0_DGP) for av in a_arr_inv]
    chioverdelta_arr = [comp_chioverdelta_DGP(av, beta_DGPv, Omega_m0_DGP, Omega_rc0_DGP) for av, beta_DGPv in zip(a_arr_inv, beta_DGP_arr)]
    Coupl_arr = [comp_coupling_DGP(beta_DGPv) for beta_DGPv in beta_DGP_arr]
    write_data_screencoupl(a_arr_inv, E_arr, E_prime_E_arr, expansion_output_filename_as_string)
    write_data_screencoupl(a_arr_inv, chioverdelta_arr, Coupl_arr, screencoupl_output_filename_as_string)

#####################################
# Cubic Galileon Barreira functions #
#####################################

# 1st Friedman eq.
def fried0_cuGalBarreira(phi_prime, E, Omm, c2, c3):
    term1 = c2*phi_prime**2./6.
    term2 = 2.*c3*E*E*phi_prime**3.
    zer = 1.-Omm-term1-term2
    return zer

# find the root of the 1st Friedman eq. to yield phi_prime
def comp_phi_prime_cuGal_Barreira(phi_prime_guess, E, Omm, c2, c3):
    phi_prime = fsolve(fried0_cuGalBarreira, phi_prime_guess, args=(E, Omm, c2, c3))[0]
    return phi_prime

def comp_E_prime_E_cuGal_Barreira(phi_prime, E, c2, c3):
    numer1 = -0.5*c2*phi_prime**2.
    numer2 = -6.*c3*E*E*phi_prime**3.*(c2+6.*c3*E*E*phi_prime)/(c2+12.*c3*E*E*phi_prime)
    numer3 = -3.
    numer = numer1+numer2+numer3
    denom = 2.+12.*c3*c3*E**4.*phi_prime**4./(c2+12.*c3*E*E*phi_prime)
    E_prime_E = numer/denom
    return E_prime_E

def comp_phi_primeprime_cuGal_Barreira(E_prime_E, phi_prime, E, c2, c3):
    numer1 = c2*(3.+E_prime_E)
    numer2 = 18.*c3*E*E*phi_prime*(1.+E_prime_E)
    denom = -c2-12.*c3*E*E*phi_prime
    phi_primeprime_phi_prime = (numer1+numer2)/denom
    return phi_primeprime_phi_prime

def comp_Omphi_Fried_cuGal_Barreira(phi_prime, E, c2, c3):
    Omega_phi = c2*phi_prime**2./6. + 2.*c3*E*E*phi_prime**3.
    return Omega_phi

def comp_alpha0_cuGal_Barreira(phi_primeprime_phi_prime, E_prime_E, phi_prime, E, c2, c3):
    term1 = -0.5*c2*phi_prime**2.
    term2 = -2.*c3*E*E*phi_prime**3.*(phi_primeprime_phi_prime+E_prime_E+2.)
    alpha0 = term1 + term2
    return alpha0

def comp_alpha1_cuGal_Barreira():
    alpha1 = 0.
    return alpha1

def comp_alpha2_cuGal_Barreira(phi_prime, E, c3):
    alpha2 = c3*E*E*phi_prime**3.
    return alpha2

def comp_beta0_cuGal_Barreira(phi_prime, E, c3):
    beta0 = c3*E*E*phi_prime**3.
    return beta0

# describes the system of ODEs
def comp_primes_cuGal_Barreira(Y, x, c2, c3):
    phi_prime, E, Omega_m = Y
    E_prime_E = comp_E_prime_E_cuGal_Barreira(phi_prime, E, c2, c3)
    E_prime = E_prime_E*E
    phi_primeprime_phi_prime = comp_phi_primeprime_cuGal_Barreira(E_prime_E, phi_prime, E, c2, c3)
    phi_primeprime = phi_primeprime_phi_prime*phi_prime
    Omega_m_prime = comp_Omega_m_prime(Omega_m, E, E_prime)
    Y_prime = [phi_primeprime, E_prime, Omega_m_prime]
    return Y_prime

def solve_inv_cuGal_Barreira(z_num, z_ini, Omega_m0, c2, c3):
    E0 = 1.
    Omega_phi0 = 1.-Omega_m0
    phi_prime_guess = 1.
    phi_prime0 = comp_phi_prime_cuGal_Barreira(phi_prime_guess, E0, Omega_m0, c2, c3)

    z_final = 0.
    x_ini = np.log(1./(1.+z_ini))
    x_final = np.log(1./(1.+z_final))
    x_arr = np.linspace(x_ini, x_final, z_num)
    a_arr = [np.exp(x) for x in x_arr]
    z_arr = [1./a - 1. for a in a_arr]
    x_arr_inv = x_arr[::-1]
    a_arr_inv = a_arr[::-1]
    z_arr_inv = z_arr[::-1]

    Y0 = [phi_prime0, E0, Omega_m0]
    Ep_E0 = comp_E_prime_E_cuGal_Barreira(phi_prime0, E0, c2, c3)
    ans = odeint(comp_primes_cuGal_Barreira, Y0, x_arr_inv, args=(c2, c3))
    phi_prime_arr = ans[:,0]
    E_arr = ans[:,1]
    Omega_m_arr = ans[:,2]

    return a_arr_inv, phi_prime_arr, E_arr, Omega_m_arr

# solves the ODEs
def run_solver_inv_cuGal_Barreira(z_num, z_ini, Omega_m0, c2, c3):
    a_arr_inv, phi_prime_arr, E_arr, Omega_m_arr = solve_inv_cuGal_Barreira(z_num, z_ini, Omega_m0, c2, c3)

    Omega_phi_arr = [1.-Ommv for Ommv in Omega_m_arr]
    Omega_phi_arr2  = [comp_Omphi_Fried_cuGal_Barreira(phi_primev, Ev, c2, c3) for phi_primev, Ev in zip(phi_prime_arr, E_arr)]
    E_prime_E_arr = [comp_E_prime_E_cuGal_Barreira(phi_primev, Ev, c2, c3) for phi_primev, Ev in zip(phi_prime_arr, E_arr)]
    E_prime_arr = [Ep_Ev*Ev for Ep_Ev, Ev in zip(E_prime_E_arr, E_arr)]
    phi_primeprime_phi_prime_arr = [comp_phi_primeprime_cuGal_Barreira(E_prime_Ev, phi_primev, Ev, c2, c3) for E_prime_Ev, phi_primev, Ev in zip(E_prime_E_arr, phi_prime_arr, E_arr)]
    Omega_m_prime_arr = [comp_Omega_m_prime(Ommv, Ev, Epv) for Ommv, Ev, Epv in zip(Omega_m_arr, E_arr, E_prime_arr)]
    alpha0_arr = [comp_alpha0_cuGal_Barreira(phi_primeprime_phi_primev, E_prime_Ev, phi_primev, Ev, c2, c3) for phi_primeprime_phi_primev, E_prime_Ev, phi_primev, Ev in zip(phi_primeprime_phi_prime_arr, E_prime_E_arr, phi_prime_arr, E_arr)]
    alpha1_arr = [comp_alpha1_cuGal_Barreira() for phi_primev, Ev in zip(phi_prime_arr, E_arr)]
    alpha2_arr = [comp_alpha2_cuGal_Barreira(phi_primev, Ev, c3) for phi_primev, Ev, in zip(phi_prime_arr, E_arr)]
    beta0_arr = [comp_beta0_cuGal_Barreira(phi_primev, Ev, c3) for phi_primev, Ev, in zip(phi_prime_arr, E_arr)]
    B_arr = [comp_B(alpha0v, alpha1v, alpha2v, beta0v) for alpha0v, alpha1v, alpha2v, beta0v in zip(alpha0_arr, alpha1_arr, alpha2_arr, beta0_arr)]
    C_arr = [comp_C(alpha0v, alpha1v, alpha2v) for alpha0v, alpha1v, alpha2v in zip(alpha0_arr, alpha1_arr, alpha2_arr)]
    return a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr

def run_solver_inv_cuGal_Barreira_bg(z_num, z_ini, Omega_m0, c2, c3):
    a_arr_inv, phi_prime_arr, E_arr, Omega_m_arr = solve_inv_cuGal_Barreira(z_num, z_ini, Omega_m0, c2, c3)

    Omega_phi_arr = [1.-Ommv for Ommv in Omega_m_arr]
    Omega_phi_arr2  = [comp_Omphi_Fried_cuGal_Barreira(phi_primev, Ev, c2, c3) for phi_primev, Ev in zip(phi_prime_arr, E_arr)]
    E_prime_E_arr = [comp_E_prime_E_cuGal_Barreira(phi_primev, Ev, c2, c3) for phi_primev, Ev in zip(phi_prime_arr, E_arr)]
    E_prime_arr = [Ep_Ev*Ev for Ep_Ev, Ev in zip(E_prime_E_arr, E_arr)]
    phi_primeprime_phi_prime_arr = [comp_phi_primeprime_cuGal_Barreira(E_prime_Ev, phi_primev, Ev, c2, c3) for E_prime_Ev, phi_primev, Ev in zip(E_prime_E_arr, phi_prime_arr, E_arr)]
    Omega_m_prime_arr = [comp_Omega_m_prime(Ommv, Ev, Epv) for Ommv, Ev, Epv in zip(Omega_m_arr, E_arr, E_prime_arr)]
    return a_arr_inv, E_arr, E_prime_E_arr, phi_prime_arr, phi_primeprime_phi_prime_arr, Omega_m_arr, Omega_m_prime_arr, Omega_phi_arr2

def run_solver_inv_cuGal_Barreira_full(z_num, z_ini, Omega_m0, c2, c3):
    a_arr_inv, phi_prime_arr, E_arr, Omega_m_arr = solve_inv_cuGal_Barreira(z_num, z_ini, Omega_m0, c2, c3)

    Omega_phi_arr = [1.-Ommv for Ommv in Omega_m_arr]
    Omega_phi_arr2  = [comp_Omphi_Fried_cuGal_Barreira(phi_primev, Ev, c2, c3) for phi_primev, Ev in zip(phi_prime_arr, E_arr)]
    E_prime_E_arr = [comp_E_prime_E_cuGal_Barreira(phi_primev, Ev, c2, c3) for phi_primev, Ev in zip(phi_prime_arr, E_arr)]
    E_prime_arr = [Ep_Ev*Ev for Ep_Ev, Ev in zip(E_prime_E_arr, E_arr)]
    phi_primeprime_phi_prime_arr = [comp_phi_primeprime_cuGal_Barreira(E_prime_Ev, phi_primev, Ev, c2, c3) for E_prime_Ev, phi_primev, Ev in zip(E_prime_E_arr, phi_prime_arr, E_arr)]
    Omega_m_prime_arr = [comp_Omega_m_prime(Ommv, Ev, Epv) for Ommv, Ev, Epv in zip(Omega_m_arr, E_arr, E_prime_arr)]
    alpha0_arr = [comp_alpha0_cuGal_Barreira(phi_primeprime_phi_primev, E_prime_Ev, phi_primev, Ev, c2, c3) for phi_primeprime_phi_primev, E_prime_Ev, phi_primev, Ev in zip(phi_primeprime_phi_prime_arr, E_prime_E_arr, phi_prime_arr, E_arr)]
    alpha1_arr = [comp_alpha1_cuGal_Barreira() for phi_primev, Ev in zip(phi_prime_arr, E_arr)]
    alpha2_arr = [comp_alpha2_cuGal_Barreira(phi_primev, Ev, c3) for phi_primev, Ev, in zip(phi_prime_arr, E_arr)]
    beta0_arr = [comp_beta0_cuGal_Barreira(phi_primev, Ev, c3) for phi_primev, Ev, in zip(phi_prime_arr, E_arr)]
    B_arr = [comp_B(alpha0v, alpha1v, alpha2v, beta0v) for alpha0v, alpha1v, alpha2v, beta0v in zip(alpha0_arr, alpha1_arr, alpha2_arr, beta0_arr)]
    C_arr = [comp_C(alpha0v, alpha1v, alpha2v) for alpha0v, alpha1v, alpha2v in zip(alpha0_arr, alpha1_arr, alpha2_arr)]
    return a_arr_inv, E_arr, E_prime_E_arr, phi_prime_arr, phi_primeprime_phi_prime_arr, Omega_m_arr, Omega_m_prime_arr, Omega_phi_arr2, alpha1_arr, alpha2_arr, B_arr, C_arr

# run the code
def runwrite_intermediate_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3, output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3)
    write_data(a_arr_inv, E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr, output_filename_as_string)

def runwrite_coupl_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3, output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    write_data_coupl(a_arr_inv, E_arr, B_arr, C_arr, Coupl_arr, output_filename_as_string)

def run_background_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3)
    return a_arr_inv, E_arr, E_prime_E_arr

def run_screencoupl_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    return a_arr_inv, chioverdelta_arr, Coupl_arr

def run_background_screencoupl_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    return a_arr_inv, E_arr, E_prime_E_arr, chioverdelta_arr, Coupl_arr

def runwrite_background_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3, expansion_output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3)
    write_data_screencoupl(a_arr_inv, E_arr, E_prime_E_arr, expansion_output_filename_as_string)

def runwrite_screencoupl_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3, output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    write_data_screencoupl(a_arr_inv, chioverdelta_arr, Coupl_arr, output_filename_as_string)

def runwrite_background_screencoupl_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3, expansion_output_filename_as_string, screencoupl_output_filename_as_string):
    a_arr_inv, E_arr, E_prime_E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr = run_solver_inv_cuGal_Barreira(z_num, z_max, Omega_m0, c2, c3)
    chioverdelta_arr = comp_chioverdelta_cuGal(a_arr_inv, E_arr, B_arr, C_arr, Omega_m0)
    Coupl_arr = comp_coupling_cuGal(alpha1_arr, alpha2_arr, C_arr)
    write_data_screencoupl(a_arr_inv, E_arr, E_prime_E_arr, expansion_output_filename_as_string)
    write_data_screencoupl(a_arr_inv, chioverdelta_arr, Coupl_arr, screencoupl_output_filename_as_string)


###################
# Growth function #
###################

# rhs = 1.5 * OmegaM * GeffOverG(a, koverH0) / (H * H * a * a * a);
#dydx[1] = rhs * D1 - (2.0 + dlogHdx) * dD1dx;

def compute_primes_MG_x(Y, x, x_arr, Omega_m_arr, E_prime_E_arr, Coupl_arr):
    """Second order differential equation for growth factor D"""
    a = np.exp(x)
    D, DD = Y
    Omega_m_spl = interp1d(x_arr, Omega_m_arr, fill_value="extrapolate")
    E_prime_E_spl = interp1d(x_arr, E_prime_E_arr, fill_value="extrapolate")
    Coupl_spl = interp1d(x_arr, Coupl_arr, fill_value="extrapolate")
    Omega_m = Omega_m_spl(x)
    E_prime_E = E_prime_E_spl(x)
    Coupl = Coupl_spl(x)
    GeffOverG = 1.+Coupl
    DDD = -1.*(2.+E_prime_E)*DD + 1.5*GeffOverG*Omega_m*D
    return [DD, DDD]

def D_DD_MG_num_x(a_arr_inv, Omega_m_arr, E_prime_E_arr, Coupl_arr):
    """Solve for growth mode"""
    x_arr_inv = [np.log(av) for av in a_arr_inv]
    x_arr = x_arr_inv[::-1]
    x_ini = np.min(x_arr)
    a_ini = np.exp(x_ini)
    z_ini = 1./a_ini - 1.
    #print('z_ini=', z_ini)
    D_ini_growth = 1. #np.exp(x_ini) #1.
    DD_ini_growth = 1. #np.exp(x_ini) #1.
    Y_ini_growth = [D_ini_growth, DD_ini_growth]
    #print(Y_ini_growth)
    ans = odeint(compute_primes_MG_x, Y_ini_growth, x_arr, args=(x_arr_inv, Omega_m_arr, E_prime_E_arr, Coupl_arr))
    D_growth_arr = ans[:,0]
    DD_growth_arr = ans[:,1]
    D_growth_arr_inv = D_growth_arr[::-1]
    DD_growth_arr_inv = DD_growth_arr[::-1]
    return D_growth_arr_inv, DD_growth_arr_inv


def compute_primes_MG_x2(Y, x, x_arr, Omega_m0, E_arr, E_prime_E_arr, Coupl_arr):
    """Second order differential equation for growth factor D"""
    a = np.exp(x)
    D, DD = Y
    E_spl = interp1d(x_arr, E_arr, fill_value="extrapolate")
    E_prime_E_spl = interp1d(x_arr, E_prime_E_arr, fill_value="extrapolate")
    Coupl_spl = interp1d(x_arr, Coupl_arr, fill_value="extrapolate")
    E = E_spl(x)
    E_prime_E = E_prime_E_spl(x)
    Coupl = Coupl_spl(x)
    Omega_m = (Omega_m0/E/E/a/a/a)
    GeffOverG = 1.+Coupl
    DDD = -1.*(2.+E_prime_E)*DD + 1.5*GeffOverG*Omega_m*D
    return [DD, DDD]

def D_DD_MG_num_x2(a_arr_inv, Omega_m0, E_arr, E_prime_E_arr, Coupl_arr):
    """Solve for growth mode"""
    x_arr_inv = [np.log(av) for av in a_arr_inv]
    x_arr = x_arr_inv[::-1]
    x_ini = np.min(x_arr)
    a_ini = np.exp(x_ini)
    z_ini = 1./a_ini - 1.
    print('z_ini=', z_ini)
    D_ini_growth = 1. #np.exp(x_ini) #1. #
    DD_ini_growth = 1. #np.exp(x_ini) #
    Y_ini_growth = [D_ini_growth, DD_ini_growth]
    print(Y_ini_growth)
    ans = odeint(compute_primes_MG_x2, Y_ini_growth, x_arr, args=(x_arr_inv, Omega_m0, E_arr, E_prime_E_arr, Coupl_arr))
    D_growth_arr = ans[:,0]
    DD_growth_arr = ans[:,1]
    D_growth_arr_inv = D_growth_arr[::-1]
    DD_growth_arr_inv = DD_growth_arr[::-1]
    return D_growth_arr_inv, DD_growth_arr_inv

def compute_primes_MG_x3(Y, x, x_arr, Omega_m0, Omega_m_arr, E_arr, E_prime_E_arr, Coupl_arr):
    """Second order differential equation for growth factor D"""
    a = np.exp(x)
    D, DD = Y
    Omega_m_spl = interp1d(x_arr, Omega_m_arr, fill_value="extrapolate")
    E_spl = interp1d(x_arr, E_arr, fill_value="extrapolate")
    E_prime_E_spl = interp1d(x_arr, E_prime_E_arr, fill_value="extrapolate")
    Coupl_spl = interp1d(x_arr, Coupl_arr, fill_value="extrapolate")
    Omega_m = Omega_m_spl(x)
    E = E_spl(x)
    E_prime_E = E_prime_E_spl(x)
    Coupl = Coupl_spl(x)
    Omega_m2 = (Omega_m0/E/E/a/a/a)
    #print(a, Omega_m, Omega_m2)
    GeffOverG = 1.+Coupl
    DDD = -1.*(2.+E_prime_E)*DD + 1.5*GeffOverG*Omega_m2*D
    return [DD, DDD]

def D_DD_MG_num_x3(a_arr_inv, Omega_m0, Omega_m_arr, E_arr, E_prime_E_arr, Coupl_arr):
    """Solve for growth mode"""
    x_arr_inv = [np.log(av) for av in a_arr_inv]
    x_arr = x_arr_inv[::-1]
    x_ini = np.min(x_arr)
    a_ini = np.exp(x_ini)
    z_ini = 1./a_ini - 1.
    #print('z_ini=', z_ini)
    D_ini_growth = 1. #np.exp(x_ini) #1. #
    DD_ini_growth = 1. #np.exp(x_ini) #
    Y_ini_growth = [D_ini_growth, DD_ini_growth]
    #print(Y_ini_growth)
    ans = odeint(compute_primes_MG_x3, Y_ini_growth, x_arr, args=(x_arr_inv, Omega_m0, Omega_m_arr, E_arr, E_prime_E_arr, Coupl_arr))
    D_growth_arr = ans[:,0]
    DD_growth_arr = ans[:,1]
    D_growth_arr_inv = D_growth_arr[::-1]
    DD_growth_arr_inv = DD_growth_arr[::-1]
    return D_growth_arr_inv, DD_growth_arr_inv

##################
# Run cuGal code #
##################

z_num = 1000
z_max = 999.
h_test = 0.7307
Omega_r0_hsq_test = 4.28e-5
Omega_b0_hsq_test = 0.02196
Omega_c0_hsq_test = 0.1274
Omega_r0_test = Omega_r0_hsq_test/h_test/h_test #0.
Omega_m0_test = (Omega_b0_hsq_test+Omega_c0_hsq_test)/h_test/h_test #0.31
E_dS_max_guess = 0.95
almost = 1e-6

def run_background_LCDMbg(z_num, z_ini, Omega_r0_LCDM, Omega_m0_LCDM):
    z_final = 0.
    x_ini = np.log(1./(1.+z_ini))
    x_final = np.log(1./(1.+z_final))
    x_arr = np.linspace(x_ini, x_final, z_num)
    a_arr = [np.exp(x) for x in x_arr]
    a_arr_inv = a_arr[::-1]
    E_arr = [comp_E_LCDM(1./av-1., Omega_r0_LCDM, Omega_m0_LCDM) for av in a_arr_inv]
    E_prime_E_arr = [comp_E_prime_E_LCDM(1./av-1., Omega_r0_LCDM, Omega_m0_LCDM) for av in a_arr_inv]
    Omega_m_arr = [comp_Omega_m_LCDM(1./av-1., Omega_r0_LCDM, Omega_m0_LCDM) for av in a_arr_inv]
    Omega_r_arr = [comp_Omega_r_LCDM(1./av-1., Omega_r0_LCDM, Omega_m0_LCDM) for av in a_arr_inv]
    Omega_L_arr = [comp_Omega_L_LCDM(1./av-1., Omega_r0_LCDM, Omega_m0_LCDM) for av in a_arr_inv]
    Coupl_arr = [0. for av in a_arr_inv]
    return a_arr_inv, E_arr, E_prime_E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, Coupl_arr

a_arr_inv_LCDMbg, E_arr_LCDMbg, E_prime_E_arr_LCDMbg, Omega_r_arr_LCDMbg, Omega_m_arr_LCDMbg, Omega_L_arr_LCDMbg, Coupl_arr_LCDMbg = run_background_LCDMbg(z_num, z_max, Omega_r0_test, Omega_m0_test)
z_arr_inv_LCDMbg = [1./av - 1. for av in a_arr_inv_LCDMbg]

a_arr_inv_f1p0, E_arr_f1p0, E_prime_E_arr_f1p0, Omega_r_arr_f1p0, Omega_m_arr_f1p0, Omega_L_arr_f1p0, Omega_phi_arr_f1p0, Omega_phi_arr2_f1p0, y_arr_f1p0 = run_solver_inv_cuGal_dS_bg(z_num, z_max, Omega_r0_test, Omega_m0_test, comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 1.0, almost), 1.0)
a_arr_inv_f0p8, E_arr_f0p8, E_prime_E_arr_f0p8, Omega_r_arr_f0p8, Omega_m_arr_f0p8, Omega_L_arr_f0p8, Omega_phi_arr_f0p8, Omega_phi_arr2_f0p8, y_arr_f0p8 = run_solver_inv_cuGal_dS_bg(z_num, z_max, Omega_r0_test, Omega_m0_test, comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.8, almost), 0.8)
a_arr_inv_f0p6, E_arr_f0p6, E_prime_E_arr_f0p6, Omega_r_arr_f0p6, Omega_m_arr_f0p6, Omega_L_arr_f0p6, Omega_phi_arr_f0p6, Omega_phi_arr2_f0p6, y_arr_f0p6 = run_solver_inv_cuGal_dS_bg(z_num, z_max, Omega_r0_test, Omega_m0_test, comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.6, almost), 0.6)
a_arr_inv_f0p4, E_arr_f0p4, E_prime_E_arr_f0p4, Omega_r_arr_f0p4, Omega_m_arr_f0p4, Omega_L_arr_f0p4, Omega_phi_arr_f0p4, Omega_phi_arr2_f0p4, y_arr_f0p4 = run_solver_inv_cuGal_dS_bg(z_num, z_max, Omega_r0_test, Omega_m0_test, comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.4, almost), 0.4)
a_arr_inv_f0p2, E_arr_f0p2, E_prime_E_arr_f0p2, Omega_r_arr_f0p2, Omega_m_arr_f0p2, Omega_L_arr_f0p2, Omega_phi_arr_f0p2, Omega_phi_arr2_f0p2, y_arr_f0p2 = run_solver_inv_cuGal_dS_bg(z_num, z_max, Omega_r0_test, Omega_m0_test, comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.2, almost), 0.2)
#a_arr_inv_f0p0, E_arr_f0p0, E_prime_E_arr_f0p0, Omega_r_arr_f0p0, Omega_m_arr_f0p0, Omega_L_arr_f0p0, Omega_phi_arr_f0p0, Omega_phi_arr2_f0p0, y_arr_f0p0 = run_solver_inv_cuGal_dS_bg(z_num, z_max, Omega_r0_test, Omega_m0_test, comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0., almost), 0.)


a_arr_inv_f1p0, E_arr_f1p0, E_prime_E_arr_f1p0, chioverdelta_arr_f1p0, Coupl_arr_f1p0 = run_background_screencoupl_cuGal_dS(z_num, z_max, Omega_r0_test, Omega_m0_test, comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 1.0, almost), 1.0)
a_arr_inv_f0p8, E_arr_f0p8, E_prime_E_arr_f0p8, chioverdelta_arr_f0p8, Coupl_arr_f0p8 = run_background_screencoupl_cuGal_dS(z_num, z_max, Omega_r0_test, Omega_m0_test, comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.8, almost), 0.8)
a_arr_inv_f0p6, E_arr_f0p6, E_prime_E_arr_f0p6, chioverdelta_arr_f0p6, Coupl_arr_f0p6 = run_background_screencoupl_cuGal_dS(z_num, z_max, Omega_r0_test, Omega_m0_test, comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.6, almost), 0.6)
a_arr_inv_f0p4, E_arr_f0p4, E_prime_E_arr_f0p4, chioverdelta_arr_f0p4, Coupl_arr_f0p4 = run_background_screencoupl_cuGal_dS(z_num, z_max, Omega_r0_test, Omega_m0_test, comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.4, almost), 0.4)
a_arr_inv_f0p2, E_arr_f0p2, E_prime_E_arr_f0p2, chioverdelta_arr_f0p2, Coupl_arr_f0p2 = run_background_screencoupl_cuGal_dS(z_num, z_max, Omega_r0_test, Omega_m0_test, comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.2, almost), 0.2)

z_arr_inv_f1p0 = [1./av-1. for av in a_arr_inv_f1p0]
z_arr_inv_f0p8 = [1./av-1. for av in a_arr_inv_f0p8]
z_arr_inv_f0p6 = [1./av-1. for av in a_arr_inv_f0p6]
z_arr_inv_f0p4 = [1./av-1. for av in a_arr_inv_f0p4]
z_arr_inv_f0p2 = [1./av-1. for av in a_arr_inv_f0p2]
#z_arr_inv_f0p0 = [1./av-1. for av in a_arr_inv_f0p0]

D_growth_arr_inv_f1p0_LCDM, DD_growth_arr_inv_f1p0_LCDM = D_DD_MG_num_x(a_arr_inv_f1p0, Omega_m_arr_LCDMbg, E_prime_E_arr_LCDMbg, np.zeros(len(Coupl_arr_f1p0),))
D_growth_arr_inv_f1p0_Omm, DD_growth_arr_inv_f1p0_Omm = D_DD_MG_num_x(a_arr_inv_f1p0, Omega_m_arr_f1p0, E_prime_E_arr_LCDMbg, np.zeros(len(Coupl_arr_f1p0),))
D_growth_arr_inv_f1p0_EpE, DD_growth_arr_inv_f1p0_EpE = D_DD_MG_num_x(a_arr_inv_f1p0, Omega_m_arr_LCDMbg, E_prime_E_arr_f1p0, np.zeros(len(Coupl_arr_f1p0),))
D_growth_arr_inv_f1p0_QCDM, DD_growth_arr_inv_f1p0_QCDM = D_DD_MG_num_x(a_arr_inv_f1p0, Omega_m_arr_f1p0, E_prime_E_arr_f1p0, np.zeros(len(Coupl_arr_f1p0),))
D_growth_arr_inv_f1p0_lincG, DD_growth_arr_inv_f1p0_lincG = D_DD_MG_num_x(a_arr_inv_f1p0, Omega_m_arr_LCDMbg, E_prime_E_arr_LCDMbg, Coupl_arr_f1p0)
D_growth_arr_inv_f1p0, DD_growth_arr_inv_f1p0 = D_DD_MG_num_x(a_arr_inv_f1p0, Omega_m_arr_f1p0, E_prime_E_arr_f1p0, Coupl_arr_f1p0)

D_growth_arr_inv_f1p0_EpE_effect = D_growth_arr_inv_f1p0_EpE/D_growth_arr_inv_f1p0_LCDM
D_growth_arr_inv_f1p0_Omm_effect = D_growth_arr_inv_f1p0_Omm/D_growth_arr_inv_f1p0_LCDM
D_growth_arr_inv_f1p0_modbg_effect = D_growth_arr_inv_f1p0_QCDM/D_growth_arr_inv_f1p0_LCDM
D_growth_arr_inv_f1p0_Geff_effect = D_growth_arr_inv_f1p0_lincG/D_growth_arr_inv_f1p0_LCDM
D_growth_arr_inv_f1p0_full_effect = D_growth_arr_inv_f1p0/D_growth_arr_inv_f1p0_LCDM
D_growth_arr_inv_f1p0_modbg_effect_zspl = interp1d(z_arr_inv_f1p0, D_growth_arr_inv_f1p0_modbg_effect, fill_value="extrapolate")

E_f1p0_modbg_effect = E_arr_f1p0/E_arr_LCDMbg
E_f0p8_modbg_effect = E_arr_f0p8/E_arr_LCDMbg
E_f0p6_modbg_effect = E_arr_f0p6/E_arr_LCDMbg
E_f0p4_modbg_effect = E_arr_f0p4/E_arr_LCDMbg
E_f0p2_modbg_effect = E_arr_f0p2/E_arr_LCDMbg

E_prime_E_f1p0_modbg_effect = np.array(E_prime_E_arr_f1p0)/np.array(E_prime_E_arr_LCDMbg)
Omega_r_f1p0_modbg_effect = Omega_r_arr_f1p0/Omega_r_arr_LCDMbg
Omega_m_f1p0_modbg_effect = Omega_m_arr_f1p0/Omega_m_arr_LCDMbg
Omega_L_f1p0_modbg_effect = Omega_L_arr_f1p0/Omega_L_arr_LCDMbg
GeffOverG_f1p0_effect = [(1.+Cv)/1. for Cv in Coupl_arr_f1p0]

dens_low = 1e-3
dens_med = 1.
dens_high = 1e3

screen_fac_arr_f1p0_low = [comp_screen_fac_Lombriser(xv, dens_low) for xv in chioverdelta_arr_f1p0]
screen_fac_arr_f1p0_med = [comp_screen_fac_Lombriser(xv, dens_med) for xv in chioverdelta_arr_f1p0]
screen_fac_arr_f1p0_high = [comp_screen_fac_Lombriser(xv, dens_high) for xv in chioverdelta_arr_f1p0]

screen_fac_arr_f0p8_low = [comp_screen_fac_Lombriser(xv, dens_low) for xv in chioverdelta_arr_f0p8]
screen_fac_arr_f0p8_med = [comp_screen_fac_Lombriser(xv, dens_med) for xv in chioverdelta_arr_f0p8]
screen_fac_arr_f0p8_high = [comp_screen_fac_Lombriser(xv, dens_high) for xv in chioverdelta_arr_f0p8]

screen_fac_arr_f0p6_low = [comp_screen_fac_Lombriser(xv, dens_low) for xv in chioverdelta_arr_f0p6]
screen_fac_arr_f0p6_med = [comp_screen_fac_Lombriser(xv, dens_med) for xv in chioverdelta_arr_f0p6]
screen_fac_arr_f0p6_high = [comp_screen_fac_Lombriser(xv, dens_high) for xv in chioverdelta_arr_f0p6]

screen_fac_arr_f0p4_low = [comp_screen_fac_Lombriser(xv, dens_low) for xv in chioverdelta_arr_f0p4]
screen_fac_arr_f0p4_med = [comp_screen_fac_Lombriser(xv, dens_med) for xv in chioverdelta_arr_f0p4]
screen_fac_arr_f0p4_high = [comp_screen_fac_Lombriser(xv, dens_high) for xv in chioverdelta_arr_f0p4]

screen_fac_arr_f0p2_low = [comp_screen_fac_Lombriser(xv, dens_low) for xv in chioverdelta_arr_f0p2]
screen_fac_arr_f0p2_med = [comp_screen_fac_Lombriser(xv, dens_med) for xv in chioverdelta_arr_f0p2]
screen_fac_arr_f0p2_high = [comp_screen_fac_Lombriser(xv, dens_high) for xv in chioverdelta_arr_f0p2]

both_arr_f1p0_low = [av*bv for av, bv in zip(screen_fac_arr_f1p0_low, Coupl_arr_f1p0)]
both_arr_f1p0_med = [av*bv for av, bv in zip(screen_fac_arr_f1p0_med, Coupl_arr_f1p0)]
both_arr_f1p0_high = [av*bv for av, bv in zip(screen_fac_arr_f1p0_high, Coupl_arr_f1p0)]

both_arr_f0p8_low = [av*bv for av, bv in zip(screen_fac_arr_f0p8_low, Coupl_arr_f0p8)]
both_arr_f0p8_med = [av*bv for av, bv in zip(screen_fac_arr_f0p8_med, Coupl_arr_f0p8)]
both_arr_f0p8_high = [av*bv for av, bv in zip(screen_fac_arr_f0p8_high, Coupl_arr_f0p8)]

both_arr_f0p6_low = [av*bv for av, bv in zip(screen_fac_arr_f0p6_low, Coupl_arr_f0p6)]
both_arr_f0p6_med = [av*bv for av, bv in zip(screen_fac_arr_f0p6_med, Coupl_arr_f0p6)]
both_arr_f0p6_high = [av*bv for av, bv in zip(screen_fac_arr_f0p6_high, Coupl_arr_f0p6)]

both_arr_f0p4_low = [av*bv for av, bv in zip(screen_fac_arr_f0p4_low, Coupl_arr_f0p4)]
both_arr_f0p4_med = [av*bv for av, bv in zip(screen_fac_arr_f0p4_med, Coupl_arr_f0p4)]
both_arr_f0p4_high = [av*bv for av, bv in zip(screen_fac_arr_f0p4_high, Coupl_arr_f0p4)]

both_arr_f0p2_low = [av*bv for av, bv in zip(screen_fac_arr_f0p2_low, Coupl_arr_f0p2)]
both_arr_f0p2_med = [av*bv for av, bv in zip(screen_fac_arr_f0p2_med, Coupl_arr_f0p2)]
both_arr_f0p2_high = [av*bv for av, bv in zip(screen_fac_arr_f0p2_high, Coupl_arr_f0p2)]

#######################
# Read cuGal sim data #
#######################

# GR
FML_GR_k_arr, FML_GR_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_z0.txt") #read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z-0.002.txt") #read_data("output/HiCOLA/pofk_new_GR_L200_Np256_Nmesh768_HiCOLA_cb_z0.000.txt") #read_data("output/HiCOLA/pofk_GR_L200_Np256_Nmesh768_HiCOLA_cb_z0.000.txt") #

# varying (f, E_dS_max(f))

FML_cG_f1p0_EdSmax_k_arr, FML_cG_f1p0_EdSmax_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_cG_f1p0_EdSmax_GR_ratio_k_arr, FML_cG_f1p0_EdSmax_GR_ratio_P_arr = comp_diff(FML_cG_f1p0_EdSmax_k_arr, FML_cG_f1p0_EdSmax_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p8_EdSmax_k_arr, FML_cG_f0p8_EdSmax_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_cG_f0p8_EdSmax_GR_ratio_k_arr, FML_cG_f0p8_EdSmax_GR_ratio_P_arr = comp_diff(FML_cG_f0p8_EdSmax_k_arr, FML_cG_f0p8_EdSmax_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p6_EdSmax_k_arr, FML_cG_f0p6_EdSmax_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_cG_f0p6_EdSmax_GR_ratio_k_arr, FML_cG_f0p6_EdSmax_GR_ratio_P_arr = comp_diff(FML_cG_f0p6_EdSmax_k_arr, FML_cG_f0p6_EdSmax_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p4_EdSmax_k_arr, FML_cG_f0p4_EdSmax_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_cG_f0p4_EdSmax_GR_ratio_k_arr, FML_cG_f0p4_EdSmax_GR_ratio_P_arr = comp_diff(FML_cG_f0p4_EdSmax_k_arr, FML_cG_f0p4_EdSmax_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p2_EdSmax_k_arr, FML_cG_f0p2_EdSmax_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_cG_f0p2_EdSmax_GR_ratio_k_arr, FML_cG_f0p2_EdSmax_GR_ratio_P_arr = comp_diff(FML_cG_f0p2_EdSmax_k_arr, FML_cG_f0p2_EdSmax_P_arr, FML_GR_k_arr, FML_GR_P_arr)

FML_cG_f1p0_EdSmax_LCDMbg_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_cG_f1p0_EdSmax_LCDMbg_GR_ratio_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_GR_ratio_P_arr = comp_diff(FML_cG_f1p0_EdSmax_LCDMbg_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p8_EdSmax_LCDMbg_k_arr, FML_cG_f0p8_EdSmax_LCDMbg_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_cG_f0p8_EdSmax_LCDMbg_GR_ratio_k_arr, FML_cG_f0p8_EdSmax_LCDMbg_GR_ratio_P_arr = comp_diff(FML_cG_f0p8_EdSmax_LCDMbg_k_arr, FML_cG_f0p8_EdSmax_LCDMbg_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p6_EdSmax_LCDMbg_k_arr, FML_cG_f0p6_EdSmax_LCDMbg_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_cG_f0p6_EdSmax_LCDMbg_GR_ratio_k_arr, FML_cG_f0p6_EdSmax_LCDMbg_GR_ratio_P_arr = comp_diff(FML_cG_f0p6_EdSmax_LCDMbg_k_arr, FML_cG_f0p6_EdSmax_LCDMbg_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p4_EdSmax_LCDMbg_k_arr, FML_cG_f0p4_EdSmax_LCDMbg_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_cG_f0p4_EdSmax_LCDMbg_GR_ratio_k_arr, FML_cG_f0p4_EdSmax_LCDMbg_GR_ratio_P_arr = comp_diff(FML_cG_f0p4_EdSmax_LCDMbg_k_arr, FML_cG_f0p4_EdSmax_LCDMbg_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p2_EdSmax_LCDMbg_k_arr, FML_cG_f0p2_EdSmax_LCDMbg_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_cG_f0p2_EdSmax_LCDMbg_GR_ratio_k_arr, FML_cG_f0p2_EdSmax_LCDMbg_GR_ratio_P_arr = comp_diff(FML_cG_f0p2_EdSmax_LCDMbg_k_arr, FML_cG_f0p2_EdSmax_LCDMbg_P_arr, FML_GR_k_arr, FML_GR_P_arr)

FML_cG_f1p0_EdSmax_unscreen_k_arr, FML_cG_f1p0_EdSmax_unscreen_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt")
FML_cG_f1p0_EdSmax_unscreen_GR_ratio_k_arr, FML_cG_f1p0_EdSmax_unscreen_GR_ratio_P_arr = comp_diff(FML_cG_f1p0_EdSmax_unscreen_k_arr, FML_cG_f1p0_EdSmax_unscreen_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p8_EdSmax_unscreen_k_arr, FML_cG_f0p8_EdSmax_unscreen_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt")
FML_cG_f0p8_EdSmax_unscreen_GR_ratio_k_arr, FML_cG_f0p8_EdSmax_unscreen_GR_ratio_P_arr = comp_diff(FML_cG_f0p8_EdSmax_unscreen_k_arr, FML_cG_f0p8_EdSmax_unscreen_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p6_EdSmax_unscreen_k_arr, FML_cG_f0p6_EdSmax_unscreen_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt")
FML_cG_f0p6_EdSmax_unscreen_GR_ratio_k_arr, FML_cG_f0p6_EdSmax_unscreen_GR_ratio_P_arr = comp_diff(FML_cG_f0p6_EdSmax_unscreen_k_arr, FML_cG_f0p6_EdSmax_unscreen_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p4_EdSmax_unscreen_k_arr, FML_cG_f0p4_EdSmax_unscreen_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt")
FML_cG_f0p4_EdSmax_unscreen_GR_ratio_k_arr, FML_cG_f0p4_EdSmax_unscreen_GR_ratio_P_arr = comp_diff(FML_cG_f0p4_EdSmax_unscreen_k_arr, FML_cG_f0p4_EdSmax_unscreen_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p2_EdSmax_unscreen_k_arr, FML_cG_f0p2_EdSmax_unscreen_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt")
FML_cG_f0p2_EdSmax_unscreen_GR_ratio_k_arr, FML_cG_f0p2_EdSmax_unscreen_GR_ratio_P_arr = comp_diff(FML_cG_f0p2_EdSmax_unscreen_k_arr, FML_cG_f0p2_EdSmax_unscreen_P_arr, FML_GR_k_arr, FML_GR_P_arr)

FML_cG_f1p0_EdSmax_GRforce_k_arr, FML_cG_f1p0_EdSmax_GRforce_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt")
FML_cG_f1p0_EdSmax_GRforce_GR_ratio_k_arr, FML_cG_f1p0_EdSmax_GRforce_GR_ratio_P_arr = comp_diff(FML_cG_f1p0_EdSmax_GRforce_k_arr, FML_cG_f1p0_EdSmax_GRforce_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p8_EdSmax_GRforce_k_arr, FML_cG_f0p8_EdSmax_GRforce_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p8_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt")
FML_cG_f0p8_EdSmax_GRforce_GR_ratio_k_arr, FML_cG_f0p8_EdSmax_GRforce_GR_ratio_P_arr = comp_diff(FML_cG_f0p8_EdSmax_GRforce_k_arr, FML_cG_f0p8_EdSmax_GRforce_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p6_EdSmax_GRforce_k_arr, FML_cG_f0p6_EdSmax_GRforce_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p6_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt")
FML_cG_f0p6_EdSmax_GRforce_GR_ratio_k_arr, FML_cG_f0p6_EdSmax_GRforce_GR_ratio_P_arr = comp_diff(FML_cG_f0p6_EdSmax_GRforce_k_arr, FML_cG_f0p6_EdSmax_GRforce_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p4_EdSmax_GRforce_k_arr, FML_cG_f0p4_EdSmax_GRforce_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p4_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt")
FML_cG_f0p4_EdSmax_GRforce_GR_ratio_k_arr, FML_cG_f0p4_EdSmax_GRforce_GR_ratio_P_arr = comp_diff(FML_cG_f0p4_EdSmax_GRforce_k_arr, FML_cG_f0p4_EdSmax_GRforce_P_arr, FML_GR_k_arr, FML_GR_P_arr)
FML_cG_f0p2_EdSmax_GRforce_k_arr, FML_cG_f0p2_EdSmax_GRforce_P_arr = read_data("output/HiCOLA/pofk_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt") #read_data("output/HiCOLA/Pyl_pofk_HiCOLA_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_z0.txt") #read_data("output/HiCOLA/pofk_cuGal_f0p2_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt")
FML_cG_f0p2_EdSmax_GRforce_GR_ratio_k_arr, FML_cG_f0p2_EdSmax_GRforce_GR_ratio_P_arr = comp_diff(FML_cG_f0p2_EdSmax_GRforce_k_arr, FML_cG_f0p2_EdSmax_GRforce_P_arr, FML_GR_k_arr, FML_GR_P_arr)

FML_cG_f1p0_EdSmax_scr5eff_ratio_k_arr, FML_cG_f1p0_EdSmax_scr5eff_ratio_P_arr = comp_diff(FML_cG_f1p0_EdSmax_k_arr, FML_cG_f1p0_EdSmax_P_arr, FML_cG_f1p0_EdSmax_GRforce_k_arr, FML_cG_f1p0_EdSmax_GRforce_P_arr)
FML_cG_f0p8_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p8_EdSmax_scr5eff_ratio_P_arr = comp_diff(FML_cG_f0p8_EdSmax_k_arr, FML_cG_f0p8_EdSmax_P_arr, FML_cG_f0p8_EdSmax_GRforce_k_arr, FML_cG_f0p8_EdSmax_GRforce_P_arr)
FML_cG_f0p6_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p6_EdSmax_scr5eff_ratio_P_arr = comp_diff(FML_cG_f0p6_EdSmax_k_arr, FML_cG_f0p6_EdSmax_P_arr, FML_cG_f0p6_EdSmax_GRforce_k_arr, FML_cG_f0p6_EdSmax_GRforce_P_arr)
FML_cG_f0p4_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p4_EdSmax_scr5eff_ratio_P_arr = comp_diff(FML_cG_f0p4_EdSmax_k_arr, FML_cG_f0p4_EdSmax_P_arr, FML_cG_f0p4_EdSmax_GRforce_k_arr, FML_cG_f0p4_EdSmax_GRforce_P_arr)
FML_cG_f0p2_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p2_EdSmax_scr5eff_ratio_P_arr = comp_diff(FML_cG_f0p2_EdSmax_k_arr, FML_cG_f0p2_EdSmax_P_arr, FML_cG_f0p2_EdSmax_GRforce_k_arr, FML_cG_f0p2_EdSmax_GRforce_P_arr)

FML_cG_f1p0_EdSmax_unscr5eff_ratio_k_arr, FML_cG_f1p0_EdSmax_unscr5eff_ratio_P_arr = comp_diff(FML_cG_f1p0_EdSmax_unscreen_k_arr, FML_cG_f1p0_EdSmax_unscreen_P_arr, FML_cG_f1p0_EdSmax_GRforce_k_arr, FML_cG_f1p0_EdSmax_GRforce_P_arr)
FML_cG_f0p8_EdSmax_unscr5eff_ratio_k_arr, FML_cG_f0p8_EdSmax_unscr5eff_ratio_P_arr = comp_diff(FML_cG_f0p8_EdSmax_unscreen_k_arr, FML_cG_f0p8_EdSmax_unscreen_P_arr, FML_cG_f0p8_EdSmax_GRforce_k_arr, FML_cG_f0p8_EdSmax_GRforce_P_arr)
FML_cG_f0p6_EdSmax_unscr5eff_ratio_k_arr, FML_cG_f0p6_EdSmax_unscr5eff_ratio_P_arr = comp_diff(FML_cG_f0p6_EdSmax_unscreen_k_arr, FML_cG_f0p6_EdSmax_unscreen_P_arr, FML_cG_f0p6_EdSmax_GRforce_k_arr, FML_cG_f0p6_EdSmax_GRforce_P_arr)
FML_cG_f0p4_EdSmax_unscr5eff_ratio_k_arr, FML_cG_f0p4_EdSmax_unscr5eff_ratio_P_arr = comp_diff(FML_cG_f0p4_EdSmax_unscreen_k_arr, FML_cG_f0p4_EdSmax_unscreen_P_arr, FML_cG_f0p4_EdSmax_GRforce_k_arr, FML_cG_f0p4_EdSmax_GRforce_P_arr)
FML_cG_f0p2_EdSmax_unscr5eff_ratio_k_arr, FML_cG_f0p2_EdSmax_unscr5eff_ratio_P_arr = comp_diff(FML_cG_f0p2_EdSmax_unscreen_k_arr, FML_cG_f0p2_EdSmax_unscreen_P_arr, FML_cG_f0p2_EdSmax_GRforce_k_arr, FML_cG_f0p2_EdSmax_GRforce_P_arr)

FML_cG_f1p0_EdSmax_modbgeff_ratio_k_arr, FML_cG_f1p0_EdSmax_modbgeff_ratio_P_arr = comp_diff(FML_cG_f1p0_EdSmax_k_arr, FML_cG_f1p0_EdSmax_P_arr, FML_cG_f1p0_EdSmax_LCDMbg_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_P_arr)
FML_cG_f0p8_EdSmax_modbgeff_ratio_k_arr, FML_cG_f0p8_EdSmax_modbgeff_ratio_P_arr = comp_diff(FML_cG_f0p8_EdSmax_k_arr, FML_cG_f0p8_EdSmax_P_arr, FML_cG_f0p8_EdSmax_LCDMbg_k_arr, FML_cG_f0p8_EdSmax_LCDMbg_P_arr)
FML_cG_f0p6_EdSmax_modbgeff_ratio_k_arr, FML_cG_f0p6_EdSmax_modbgeff_ratio_P_arr = comp_diff(FML_cG_f0p6_EdSmax_k_arr, FML_cG_f0p6_EdSmax_P_arr, FML_cG_f0p6_EdSmax_LCDMbg_k_arr, FML_cG_f0p6_EdSmax_LCDMbg_P_arr)
FML_cG_f0p4_EdSmax_modbgeff_ratio_k_arr, FML_cG_f0p4_EdSmax_modbgeff_ratio_P_arr = comp_diff(FML_cG_f0p4_EdSmax_k_arr, FML_cG_f0p4_EdSmax_P_arr, FML_cG_f0p4_EdSmax_LCDMbg_k_arr, FML_cG_f0p4_EdSmax_LCDMbg_P_arr)
FML_cG_f0p2_EdSmax_modbgeff_ratio_k_arr, FML_cG_f0p2_EdSmax_modbgeff_ratio_P_arr = comp_diff(FML_cG_f0p2_EdSmax_k_arr, FML_cG_f0p2_EdSmax_P_arr, FML_cG_f0p2_EdSmax_LCDMbg_k_arr, FML_cG_f0p2_EdSmax_LCDMbg_P_arr)

FML_cG_f1p0_EdSmax_screeneff_ratio_k_arr, FML_cG_f1p0_EdSmax_screeneff_ratio_P_arr = comp_diff(FML_cG_f1p0_EdSmax_k_arr, FML_cG_f1p0_EdSmax_P_arr, FML_cG_f1p0_EdSmax_unscreen_k_arr, FML_cG_f1p0_EdSmax_unscreen_P_arr)
FML_cG_f0p8_EdSmax_screeneff_ratio_k_arr, FML_cG_f0p8_EdSmax_screeneff_ratio_P_arr = comp_diff(FML_cG_f0p8_EdSmax_k_arr, FML_cG_f0p8_EdSmax_P_arr, FML_cG_f0p8_EdSmax_unscreen_k_arr, FML_cG_f0p8_EdSmax_unscreen_P_arr)
FML_cG_f0p6_EdSmax_screeneff_ratio_k_arr, FML_cG_f0p6_EdSmax_screeneff_ratio_P_arr = comp_diff(FML_cG_f0p6_EdSmax_k_arr, FML_cG_f0p6_EdSmax_P_arr, FML_cG_f0p6_EdSmax_unscreen_k_arr, FML_cG_f0p6_EdSmax_unscreen_P_arr)
FML_cG_f0p4_EdSmax_screeneff_ratio_k_arr, FML_cG_f0p4_EdSmax_screeneff_ratio_P_arr = comp_diff(FML_cG_f0p4_EdSmax_k_arr, FML_cG_f0p4_EdSmax_P_arr, FML_cG_f0p4_EdSmax_unscreen_k_arr, FML_cG_f0p4_EdSmax_unscreen_P_arr)
FML_cG_f0p2_EdSmax_screeneff_ratio_k_arr, FML_cG_f0p2_EdSmax_screeneff_ratio_P_arr = comp_diff(FML_cG_f0p2_EdSmax_k_arr, FML_cG_f0p2_EdSmax_P_arr, FML_cG_f0p2_EdSmax_unscreen_k_arr, FML_cG_f0p2_EdSmax_unscreen_P_arr)

# evolution with z
FML_cG_f1p0_EdSmax_z0p109_k_arr, FML_cG_f1p0_EdSmax_z0p109_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z0.109.txt")
FML_cG_f1p0_EdSmax_LCDMbg_z0p109_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_z0p109_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z0.109.txt")
FML_cG_f1p0_EdSmax_unscreen_z0p109_k_arr, FML_cG_f1p0_EdSmax_unscreen_z0p109_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z0.109.txt")
FML_cG_f1p0_EdSmax_GRforce_z0p109_k_arr, FML_cG_f1p0_EdSmax_GRforce_z0p109_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z0.109.txt")
FML_cG_f1p0_EdSmax_modbgeff_ratio_z0p109_k_arr, FML_cG_f1p0_EdSmax_modbgeff_ratio_z0p109_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z0p109_k_arr, FML_cG_f1p0_EdSmax_z0p109_P_arr, FML_cG_f1p0_EdSmax_LCDMbg_z0p109_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_z0p109_P_arr)
FML_cG_f1p0_EdSmax_unscr5eff_ratio_z0p109_k_arr, FML_cG_f1p0_EdSmax_unscr5eff_ratio_z0p109_P_arr = comp_diff(FML_cG_f1p0_EdSmax_unscreen_z0p109_k_arr, FML_cG_f1p0_EdSmax_unscreen_z0p109_P_arr, FML_cG_f1p0_EdSmax_GRforce_z0p109_k_arr, FML_cG_f1p0_EdSmax_GRforce_z0p109_P_arr)
FML_cG_f1p0_EdSmax_screeneff_ratio_z0p109_k_arr, FML_cG_f1p0_EdSmax_screeneff_ratio_z0p109_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z0p109_k_arr, FML_cG_f1p0_EdSmax_z0p109_P_arr, FML_cG_f1p0_EdSmax_unscreen_z0p109_k_arr, FML_cG_f1p0_EdSmax_unscreen_z0p109_P_arr)
FML_cG_f1p0_EdSmax_scr5eff_ratio_z0p109_k_arr, FML_cG_f1p0_EdSmax_scr5eff_ratio_z0p109_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z0p109_k_arr, FML_cG_f1p0_EdSmax_z0p109_P_arr, FML_cG_f1p0_EdSmax_GRforce_z0p109_k_arr, FML_cG_f1p0_EdSmax_GRforce_z0p109_P_arr)

FML_cG_f1p0_EdSmax_z0p480_k_arr, FML_cG_f1p0_EdSmax_z0p480_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z0.480.txt")
FML_cG_f1p0_EdSmax_LCDMbg_z0p480_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_z0p480_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z0.480.txt")
FML_cG_f1p0_EdSmax_unscreen_z0p480_k_arr, FML_cG_f1p0_EdSmax_unscreen_z0p480_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z0.480.txt")
FML_cG_f1p0_EdSmax_GRforce_z0p480_k_arr, FML_cG_f1p0_EdSmax_GRforce_z0p480_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z0.480.txt")
FML_cG_f1p0_EdSmax_modbgeff_ratio_z0p480_k_arr, FML_cG_f1p0_EdSmax_modbgeff_ratio_z0p480_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z0p480_k_arr, FML_cG_f1p0_EdSmax_z0p480_P_arr, FML_cG_f1p0_EdSmax_LCDMbg_z0p480_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_z0p480_P_arr)
FML_cG_f1p0_EdSmax_unscr5eff_ratio_z0p480_k_arr, FML_cG_f1p0_EdSmax_unscr5eff_ratio_z0p480_P_arr = comp_diff(FML_cG_f1p0_EdSmax_unscreen_z0p480_k_arr, FML_cG_f1p0_EdSmax_unscreen_z0p480_P_arr, FML_cG_f1p0_EdSmax_GRforce_z0p480_k_arr, FML_cG_f1p0_EdSmax_GRforce_z0p480_P_arr)
FML_cG_f1p0_EdSmax_screeneff_ratio_z0p480_k_arr, FML_cG_f1p0_EdSmax_screeneff_ratio_z0p480_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z0p480_k_arr, FML_cG_f1p0_EdSmax_z0p480_P_arr, FML_cG_f1p0_EdSmax_unscreen_z0p480_k_arr, FML_cG_f1p0_EdSmax_unscreen_z0p480_P_arr)
FML_cG_f1p0_EdSmax_scr5eff_ratio_z0p480_k_arr, FML_cG_f1p0_EdSmax_scr5eff_ratio_z0p480_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z0p480_k_arr, FML_cG_f1p0_EdSmax_z0p480_P_arr, FML_cG_f1p0_EdSmax_GRforce_z0p480_k_arr, FML_cG_f1p0_EdSmax_GRforce_z0p480_P_arr)

FML_cG_f1p0_EdSmax_z0p985_k_arr, FML_cG_f1p0_EdSmax_z0p985_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z0.985.txt")
FML_cG_f1p0_EdSmax_LCDMbg_z0p985_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_z0p985_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z0.985.txt")
FML_cG_f1p0_EdSmax_unscreen_z0p985_k_arr, FML_cG_f1p0_EdSmax_unscreen_z0p985_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z0.985.txt")
FML_cG_f1p0_EdSmax_GRforce_z0p985_k_arr, FML_cG_f1p0_EdSmax_GRforce_z0p985_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z0.985.txt")
FML_cG_f1p0_EdSmax_modbgeff_ratio_z0p985_k_arr, FML_cG_f1p0_EdSmax_modbgeff_ratio_z0p985_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z0p985_k_arr, FML_cG_f1p0_EdSmax_z0p985_P_arr, FML_cG_f1p0_EdSmax_LCDMbg_z0p985_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_z0p985_P_arr)
FML_cG_f1p0_EdSmax_unscr5eff_ratio_z0p985_k_arr, FML_cG_f1p0_EdSmax_unscr5eff_ratio_z0p985_P_arr = comp_diff(FML_cG_f1p0_EdSmax_unscreen_z0p985_k_arr, FML_cG_f1p0_EdSmax_unscreen_z0p985_P_arr, FML_cG_f1p0_EdSmax_GRforce_z0p985_k_arr, FML_cG_f1p0_EdSmax_GRforce_z0p985_P_arr)
FML_cG_f1p0_EdSmax_screeneff_ratio_z0p985_k_arr, FML_cG_f1p0_EdSmax_screeneff_ratio_z0p985_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z0p985_k_arr, FML_cG_f1p0_EdSmax_z0p985_P_arr, FML_cG_f1p0_EdSmax_unscreen_z0p985_k_arr, FML_cG_f1p0_EdSmax_unscreen_z0p985_P_arr)
FML_cG_f1p0_EdSmax_scr5eff_ratio_z0p985_k_arr, FML_cG_f1p0_EdSmax_scr5eff_ratio_z0p985_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z0p985_k_arr, FML_cG_f1p0_EdSmax_z0p985_P_arr, FML_cG_f1p0_EdSmax_GRforce_z0p985_k_arr, FML_cG_f1p0_EdSmax_GRforce_z0p985_P_arr)

FML_cG_f1p0_EdSmax_z1p990_k_arr, FML_cG_f1p0_EdSmax_z1p990_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z1.990.txt")
FML_cG_f1p0_EdSmax_LCDMbg_z1p990_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_z1p990_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z1.990.txt")
FML_cG_f1p0_EdSmax_unscreen_z1p990_k_arr, FML_cG_f1p0_EdSmax_unscreen_z1p990_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z1.990.txt")
FML_cG_f1p0_EdSmax_GRforce_z1p990_k_arr, FML_cG_f1p0_EdSmax_GRforce_z1p990_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z1.990.txt")
FML_cG_f1p0_EdSmax_modbgeff_ratio_z1p990_k_arr, FML_cG_f1p0_EdSmax_modbgeff_ratio_z1p990_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z1p990_k_arr, FML_cG_f1p0_EdSmax_z1p990_P_arr, FML_cG_f1p0_EdSmax_LCDMbg_z1p990_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_z1p990_P_arr)
FML_cG_f1p0_EdSmax_unscr5eff_ratio_z1p990_k_arr, FML_cG_f1p0_EdSmax_unscr5eff_ratio_z1p990_P_arr = comp_diff(FML_cG_f1p0_EdSmax_unscreen_z1p990_k_arr, FML_cG_f1p0_EdSmax_unscreen_z1p990_P_arr, FML_cG_f1p0_EdSmax_GRforce_z1p990_k_arr, FML_cG_f1p0_EdSmax_GRforce_z1p990_P_arr)
FML_cG_f1p0_EdSmax_screeneff_ratio_z1p990_k_arr, FML_cG_f1p0_EdSmax_screeneff_ratio_z1p990_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z1p990_k_arr, FML_cG_f1p0_EdSmax_z1p990_P_arr, FML_cG_f1p0_EdSmax_unscreen_z1p990_k_arr, FML_cG_f1p0_EdSmax_unscreen_z1p990_P_arr)
FML_cG_f1p0_EdSmax_scr5eff_ratio_z1p990_k_arr, FML_cG_f1p0_EdSmax_scr5eff_ratio_z1p990_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z1p990_k_arr, FML_cG_f1p0_EdSmax_z1p990_P_arr, FML_cG_f1p0_EdSmax_GRforce_z1p990_k_arr, FML_cG_f1p0_EdSmax_GRforce_z1p990_P_arr)

FML_cG_f1p0_EdSmax_z9p804_k_arr, FML_cG_f1p0_EdSmax_z9p804_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z9.804.txt")
FML_cG_f1p0_EdSmax_LCDMbg_z9p804_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_z9p804_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_LCDMbg_lin2LPT_screen_forcelink02_smoothR3_cb_z9.804.txt")
FML_cG_f1p0_EdSmax_unscreen_z9p804_k_arr, FML_cG_f1p0_EdSmax_unscreen_z9p804_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z9.804.txt")
FML_cG_f1p0_EdSmax_GRforce_z9p804_k_arr, FML_cG_f1p0_EdSmax_GRforce_z9p804_P_arr = read_data("output/HiCOLA/pofk_cuGal_f1p0_EdS_max_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z9.804.txt")
FML_cG_f1p0_EdSmax_modbgeff_ratio_z9p804_k_arr, FML_cG_f1p0_EdSmax_modbgeff_ratio_z9p804_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z9p804_k_arr, FML_cG_f1p0_EdSmax_z9p804_P_arr, FML_cG_f1p0_EdSmax_LCDMbg_z9p804_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_z9p804_P_arr)
FML_cG_f1p0_EdSmax_unscr5eff_ratio_z9p804_k_arr, FML_cG_f1p0_EdSmax_unscr5eff_ratio_z9p804_P_arr = comp_diff(FML_cG_f1p0_EdSmax_unscreen_z9p804_k_arr, FML_cG_f1p0_EdSmax_unscreen_z9p804_P_arr, FML_cG_f1p0_EdSmax_GRforce_z9p804_k_arr, FML_cG_f1p0_EdSmax_GRforce_z9p804_P_arr)
FML_cG_f1p0_EdSmax_screeneff_ratio_z9p804_k_arr, FML_cG_f1p0_EdSmax_screeneff_ratio_z9p804_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z9p804_k_arr, FML_cG_f1p0_EdSmax_z9p804_P_arr, FML_cG_f1p0_EdSmax_unscreen_z9p804_k_arr, FML_cG_f1p0_EdSmax_unscreen_z9p804_P_arr)
FML_cG_f1p0_EdSmax_scr5eff_ratio_z9p804_k_arr, FML_cG_f1p0_EdSmax_scr5eff_ratio_z9p804_P_arr = comp_diff(FML_cG_f1p0_EdSmax_z9p804_k_arr, FML_cG_f1p0_EdSmax_z9p804_P_arr, FML_cG_f1p0_EdSmax_GRforce_z9p804_k_arr, FML_cG_f1p0_EdSmax_GRforce_z9p804_P_arr)


######################
# Read Traykova data #
######################

def read_Ashim_bg(input_filename_as_string):
    data = np.loadtxt(input_filename_as_string)
    a_arr = data[:,0]
    E_arr = data[:,1]
    E_prime_E_arr = data[:,2]
    Omega_r_arr = data[:,3]
    Omega_m_arr = data[:,4]
    Omega_L_arr = data[:,5]
    Omega_phi_arr = data[:,6]
    y_arr = data[:,7]
    return a_arr, E_arr, E_prime_E_arr, Omega_r_arr, Omega_m_arr, Omega_L_arr, Omega_phi_arr, y_arr

# Ashim's input files
TraykovaL2_a_arr, TraykovaL2_E_arr, TraykovaL2_E_prime_E_arr, TraykovaL2_Omega_r_arr, TraykovaL2_Omega_m_arr, TraykovaL2_Omega_L_arr, TraykovaL2_Omega_phi_arr, TraykovaL2_y_arr = read_Ashim_bg("HiCOLA_input/L2_full_expansion_history.txt")
TraykovaL2L11_a_arr, TraykovaL2L11_E_arr, TraykovaL2L11_E_prime_E_arr, TraykovaL2L11_Omega_r_arr, TraykovaL2L11_Omega_m_arr, TraykovaL2L11_Omega_L_arr, TraykovaL2L11_Omega_phi_arr, TraykovaL2L11_y_arr = read_Ashim_bg("HiCOLA_input/L2-L11_full_expansion_history.txt")
TraykovaL2L39_a_arr, TraykovaL2L39_E_arr, TraykovaL2L39_E_prime_E_arr, TraykovaL2L39_Omega_r_arr, TraykovaL2L39_Omega_m_arr, TraykovaL2L39_Omega_L_arr, TraykovaL2L39_Omega_phi_arr, TraykovaL2L39_y_arr = read_Ashim_bg("HiCOLA_input/L2-L39_full_expansion_history.txt")

TraykovaL2_z_arr_inv = [1./av-1. for av in TraykovaL2_a_arr]
TraykovaL2L11_z_arr_inv = [1./av-1. for av in TraykovaL2L11_a_arr]
TraykovaL2L39_z_arr_inv = [1./av-1. for av in TraykovaL2L39_a_arr]

TraykovaL2_a_exp_arr, TraykovaL2_E_arr2, TraykovaL2_E_prime_E_arr2 = read_data3("HiCOLA_input/Traykova_expansion_L2.txt")
TraykovaL2_a_scr_arr, TraykovaL2_chioverdelta_arr, TraykovaL2_coupl_arr = read_data3("HiCOLA_input/Traykova_ChiCoupl_L2.txt")

TraykovaL2L11_a_exp_arr, TraykovaL2L11_E_arr2, TraykovaL2L11_E_prime_E_arr2 = read_data3("HiCOLA_input/Traykova_expansion_L2-L11.txt")
TraykovaL2L11_a_scr_arr, TraykovaL2L11_chioverdelta_arr, TraykovaL2L11_coupl_arr = read_data3("HiCOLA_input/Traykova_ChiCoupl_L2-L11.txt")

TraykovaL2L39_a_exp_arr, TraykovaL2L39_E_arr2, TraykovaL2L39_E_prime_E_arr2 = read_data3("HiCOLA_input/Traykova_expansion_L2-L39.txt")
TraykovaL2L39_a_scr_arr, TraykovaL2L39_chioverdelta_arr, TraykovaL2L39_coupl_arr = read_data3("HiCOLA_input/Traykova_ChiCoupl_L2-L39.txt")

TraykovaL2_z_arr = [1./av-1. for av in TraykovaL2_a_scr_arr]
TraykovaL2L11_z_arr = [1./av-1. for av in TraykovaL2L11_a_scr_arr]
TraykovaL2L39_z_arr = [1./av-1. for av in TraykovaL2L39_a_scr_arr]

TraykovaL2_interp_E = interp1d(TraykovaL2_z_arr, TraykovaL2_E_arr2, fill_value="extrapolate")
TraykovaL2_interp_E_arr = TraykovaL2_interp_E(z_arr_inv_LCDMbg)
TraykovaL2_interp_E_prime_E = interp1d(TraykovaL2_z_arr, TraykovaL2_E_prime_E_arr2, fill_value="extrapolate")
TraykovaL2_interp_E_prime_E_arr = TraykovaL2_interp_E_prime_E(z_arr_inv_LCDMbg)
TraykovaL2_interp_coupl = interp1d(TraykovaL2_z_arr, TraykovaL2_coupl_arr, fill_value="extrapolate")
TraykovaL2_interp_coupl_arr = TraykovaL2_interp_coupl(z_arr_inv_LCDMbg)
E_TraykovaL2_modbg_effect = TraykovaL2_interp_E_arr/E_arr_LCDMbg
E_prime_E_TraykovaL2_modbg_effect = np.array(TraykovaL2_interp_E_prime_E_arr)/np.array(E_prime_E_arr_LCDMbg)
GeffOverG_TraykovaL2_effect = [(1.+Cv)/1. for Cv in TraykovaL2_interp_coupl_arr]

TraykovaL2L11_interp_E = interp1d(TraykovaL2L11_z_arr, TraykovaL2L11_E_arr2, fill_value="extrapolate")
TraykovaL2L11_interp_E_arr = TraykovaL2L11_interp_E(z_arr_inv_LCDMbg)
TraykovaL2L11_interp_E_prime_E = interp1d(TraykovaL2L11_z_arr, TraykovaL2L11_E_prime_E_arr2, fill_value="extrapolate")
TraykovaL2L11_interp_E_prime_E_arr = TraykovaL2L11_interp_E_prime_E(z_arr_inv_LCDMbg)
TraykovaL2L11_interp_coupl = interp1d(TraykovaL2L11_z_arr, TraykovaL2L11_coupl_arr, fill_value="extrapolate")
TraykovaL2L11_interp_coupl_arr = TraykovaL2L11_interp_coupl(z_arr_inv_LCDMbg)
E_TraykovaL2L11_modbg_effect = TraykovaL2L11_interp_E_arr/E_arr_LCDMbg
E_prime_E_TraykovaL2L11_modbg_effect = np.array(TraykovaL2L11_interp_E_prime_E_arr)/np.array(E_prime_E_arr_LCDMbg)
GeffOverG_TraykovaL2L11_effect = [(1.+Cv)/1. for Cv in TraykovaL2L11_interp_coupl_arr]

TraykovaL2L39_interp_E = interp1d(TraykovaL2L39_z_arr, TraykovaL2L39_E_arr2, fill_value="extrapolate")
TraykovaL2L39_interp_E_arr = TraykovaL2L39_interp_E(z_arr_inv_LCDMbg)
TraykovaL2L39_interp_E_prime_E = interp1d(TraykovaL2L39_z_arr, TraykovaL2L39_E_prime_E_arr2, fill_value="extrapolate")
TraykovaL2L39_interp_E_prime_E_arr = TraykovaL2L39_interp_E_prime_E(z_arr_inv_LCDMbg)
TraykovaL2L39_interp_Omega_m = interp1d(TraykovaL2L39_z_arr, TraykovaL2L39_Omega_m_arr, fill_value="extrapolate")
TraykovaL2L39_interp_Omega_m_arr = TraykovaL2L39_interp_Omega_m(z_arr_inv_LCDMbg)
TraykovaL2L39_interp_coupl = interp1d(TraykovaL2L39_z_arr, TraykovaL2L39_coupl_arr, fill_value="extrapolate")
TraykovaL2L39_interp_coupl_arr = TraykovaL2L39_interp_coupl(z_arr_inv_LCDMbg)
E_TraykovaL2L39_modbg_effect = TraykovaL2L39_interp_E_arr/E_arr_LCDMbg
E_prime_E_TraykovaL2L39_modbg_effect = np.array(TraykovaL2L39_interp_E_prime_E_arr)/np.array(E_prime_E_arr_LCDMbg)
GeffOverG_TraykovaL2L39_effect = [(1.+Cv)/1. for Cv in TraykovaL2L39_interp_coupl_arr]

D_growth_arr_inv_TraykovaL2L39_LCDM, DD_growth_arr_inv_TraykovaL2L39_LCDM = D_DD_MG_num_x(a_arr_inv_LCDMbg, Omega_m_arr_LCDMbg, E_prime_E_arr_LCDMbg, np.zeros(len(TraykovaL2L39_interp_coupl_arr),))
D_growth_arr_inv_TraykovaL2L39_Omm, DD_growth_arr_inv_TraykovaL2L39_Omm = D_DD_MG_num_x(a_arr_inv_LCDMbg, TraykovaL2L39_interp_Omega_m_arr, E_prime_E_arr_LCDMbg, np.zeros(len(TraykovaL2L39_interp_coupl_arr),))
D_growth_arr_inv_TraykovaL2L39_EpE, DD_growth_arr_inv_TraykovaL2L39_EpE = D_DD_MG_num_x(a_arr_inv_LCDMbg, Omega_m_arr_LCDMbg, TraykovaL2L39_interp_E_prime_E_arr, np.zeros(len(TraykovaL2L39_interp_coupl_arr),))
D_growth_arr_inv_TraykovaL2L39_QCDM, DD_growth_arr_inv_TraykovaL2L39_QCDM = D_DD_MG_num_x(a_arr_inv_LCDMbg, TraykovaL2L39_interp_Omega_m_arr, TraykovaL2L39_interp_E_prime_E_arr, np.zeros(len(TraykovaL2L39_interp_coupl_arr),))
D_growth_arr_inv_TraykovaL2L39_lincG, DD_growth_arr_inv_TraykovaL2L39_lincG = D_DD_MG_num_x(a_arr_inv_LCDMbg, Omega_m_arr_LCDMbg, E_prime_E_arr_LCDMbg, TraykovaL2L39_interp_coupl_arr)
D_growth_arr_inv_TraykovaL2L39, DD_growth_arr_inv_TraykovaL2L39 = D_DD_MG_num_x(a_arr_inv_LCDMbg, TraykovaL2L39_interp_Omega_m_arr, TraykovaL2L39_interp_E_prime_E_arr, TraykovaL2L39_interp_coupl_arr)

D_growth_arr_inv_TraykovaL2L39_EpE_effect = D_growth_arr_inv_TraykovaL2L39_EpE/D_growth_arr_inv_TraykovaL2L39_LCDM
D_growth_arr_inv_TraykovaL2L39_Omm_effect = D_growth_arr_inv_TraykovaL2L39_Omm/D_growth_arr_inv_TraykovaL2L39_LCDM
D_growth_arr_inv_TraykovaL2L39_modbg_effect = D_growth_arr_inv_TraykovaL2L39_QCDM/D_growth_arr_inv_TraykovaL2L39_LCDM
D_growth_arr_inv_TraykovaL2L39_Geff_effect = D_growth_arr_inv_TraykovaL2L39_lincG/D_growth_arr_inv_TraykovaL2L39_LCDM
D_growth_arr_inv_TraykovaL2L39_full_effect = D_growth_arr_inv_TraykovaL2L39/D_growth_arr_inv_TraykovaL2L39_LCDM

# screening factors
#dens_low = 1e-3
#dens_med = 1e0
#dens_high = 1e3

screen_fac_arr_TraykovaL2_low = [comp_screen_fac_Lombriser(xv, dens_low) for xv in TraykovaL2_chioverdelta_arr]
screen_fac_arr_TraykovaL2_med = [comp_screen_fac_Lombriser(xv, dens_med) for xv in TraykovaL2_chioverdelta_arr]
screen_fac_arr_TraykovaL2_high = [comp_screen_fac_Lombriser(xv, dens_high) for xv in TraykovaL2_chioverdelta_arr]

screen_fac_arr_TraykovaL2L11_low = [comp_screen_fac_Lombriser(xv, dens_low) for xv in TraykovaL2L11_chioverdelta_arr]
screen_fac_arr_TraykovaL2L11_med = [comp_screen_fac_Lombriser(xv, dens_med) for xv in TraykovaL2L11_chioverdelta_arr]
screen_fac_arr_TraykovaL2L11_high = [comp_screen_fac_Lombriser(xv, dens_high) for xv in TraykovaL2L11_chioverdelta_arr]

screen_fac_arr_TraykovaL2L39_low = [comp_screen_fac_Lombriser(xv, dens_low) for xv in TraykovaL2L39_chioverdelta_arr]
screen_fac_arr_TraykovaL2L39_med = [comp_screen_fac_Lombriser(xv, dens_med) for xv in TraykovaL2L39_chioverdelta_arr]
screen_fac_arr_TraykovaL2L39_high = [comp_screen_fac_Lombriser(xv, dens_high) for xv in TraykovaL2L39_chioverdelta_arr]

# screen_fac x coupling
both_arr_TraykovaL2_low = [av*bv for av, bv in zip(screen_fac_arr_TraykovaL2_low, TraykovaL2_coupl_arr)]
both_arr_TraykovaL2_med = [av*bv for av, bv in zip(screen_fac_arr_TraykovaL2_med, TraykovaL2_coupl_arr)]
both_arr_TraykovaL2_high = [av*bv for av, bv in zip(screen_fac_arr_TraykovaL2_high, TraykovaL2_coupl_arr)]

both_arr_TraykovaL2L11_low = [av*bv for av, bv in zip(screen_fac_arr_TraykovaL2L11_low, TraykovaL2L11_coupl_arr)]
both_arr_TraykovaL2L11_med = [av*bv for av, bv in zip(screen_fac_arr_TraykovaL2L11_med, TraykovaL2L11_coupl_arr)]
both_arr_TraykovaL2L11_high = [av*bv for av, bv in zip(screen_fac_arr_TraykovaL2L11_high, TraykovaL2L11_coupl_arr)]

both_arr_TraykovaL2L39_low = [av*bv for av, bv in zip(screen_fac_arr_TraykovaL2L39_low, TraykovaL2L39_coupl_arr)]
both_arr_TraykovaL2L39_med = [av*bv for av, bv in zip(screen_fac_arr_TraykovaL2L39_med, TraykovaL2L39_coupl_arr)]
both_arr_TraykovaL2L39_high = [av*bv for av, bv in zip(screen_fac_arr_TraykovaL2L39_high, TraykovaL2L39_coupl_arr)]

# power spectra
FML_GR_L400_Np512_Nmesh1536_z0_k_arr, FML_GR_L400_Np512_Nmesh1536_z0_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z-0.002.txt") #"output/HiCOLA/Pyl_pofk_HiCOLA_GR_L400_Np512_Nmesh1536_HiCOLA_z0.txt")

'''
# Investigation of power transfer in LCDM
FML_GR_L400_Np512_Nmesh1536_z0109_k_arr, FML_GR_L400_Np512_Nmesh1536_z0109_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z0.109.txt")
FML_GR_L400_Np512_Nmesh1536_z0480_k_arr, FML_GR_L400_Np512_Nmesh1536_z0480_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z0.480.txt")
FML_GR_L400_Np512_Nmesh1536_z0985_k_arr, FML_GR_L400_Np512_Nmesh1536_z0985_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z0.985.txt")
FML_GR_L400_Np512_Nmesh1536_z1990_k_arr, FML_GR_L400_Np512_Nmesh1536_z1990_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z1.990.txt")
FML_GR_L400_Np512_Nmesh1536_z48999_k_arr, FML_GR_L400_Np512_Nmesh1536_z48999_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z48.999.txt")

FML_GR_L400_Np512_Nmesh1536_z0_Pgrowth_arr = [x/y for x,y in zip(FML_GR_L400_Np512_Nmesh1536_z0_P_arr, FML_GR_L400_Np512_Nmesh1536_z48999_P_arr)]
FML_GR_L400_Np512_Nmesh1536_z0109_Pgrowth_arr = [x/y for x,y in zip(FML_GR_L400_Np512_Nmesh1536_z0109_P_arr, FML_GR_L400_Np512_Nmesh1536_z48999_P_arr)]
FML_GR_L400_Np512_Nmesh1536_z0480_Pgrowth_arr = [x/y for x,y in zip(FML_GR_L400_Np512_Nmesh1536_z0480_P_arr, FML_GR_L400_Np512_Nmesh1536_z48999_P_arr)]
FML_GR_L400_Np512_Nmesh1536_z0985_Pgrowth_arr = [x/y for x,y in zip(FML_GR_L400_Np512_Nmesh1536_z0985_P_arr, FML_GR_L400_Np512_Nmesh1536_z48999_P_arr)]
FML_GR_L400_Np512_Nmesh1536_z1990_Pgrowth_arr = [x/y for x,y in zip(FML_GR_L400_Np512_Nmesh1536_z1990_P_arr, FML_GR_L400_Np512_Nmesh1536_z48999_P_arr)]

fig = plt.figure()
plt.semilogx(FML_GR_L400_Np512_Nmesh1536_z1990_k_arr, FML_GR_L400_Np512_Nmesh1536_z1990_Pgrowth_arr, label='z=1.990')
plt.semilogx(FML_GR_L400_Np512_Nmesh1536_z0985_k_arr, FML_GR_L400_Np512_Nmesh1536_z0985_Pgrowth_arr, label='z=0.985')
plt.semilogx(FML_GR_L400_Np512_Nmesh1536_z0480_k_arr, FML_GR_L400_Np512_Nmesh1536_z0480_Pgrowth_arr, label='z=0.480')
plt.semilogx(FML_GR_L400_Np512_Nmesh1536_z0109_k_arr, FML_GR_L400_Np512_Nmesh1536_z0109_Pgrowth_arr, label='z=0.109')
plt.semilogx(FML_GR_L400_Np512_Nmesh1536_z0_k_arr, FML_GR_L400_Np512_Nmesh1536_z0_Pgrowth_arr, label='z=0.000')
plt.xlim(0.02, 2.)
plt.legend()
plt.show()


fig = plt.figure()
plt.loglog(FML_GR_L400_Np512_Nmesh1536_z48999_k_arr, FML_GR_L400_Np512_Nmesh1536_z48999_P_arr, label='z=48.999')
plt.loglog(FML_GR_L400_Np512_Nmesh1536_z1990_k_arr, FML_GR_L400_Np512_Nmesh1536_z1990_P_arr, label='z=1.990')
plt.loglog(FML_GR_L400_Np512_Nmesh1536_z0985_k_arr, FML_GR_L400_Np512_Nmesh1536_z0985_P_arr, label='z=0.985')
plt.loglog(FML_GR_L400_Np512_Nmesh1536_z0480_k_arr, FML_GR_L400_Np512_Nmesh1536_z0480_P_arr, label='z=0.480')
plt.loglog(FML_GR_L400_Np512_Nmesh1536_z0109_k_arr, FML_GR_L400_Np512_Nmesh1536_z0109_P_arr, label='z=0.109')
plt.loglog(FML_GR_L400_Np512_Nmesh1536_z0_k_arr, FML_GR_L400_Np512_Nmesh1536_z0_P_arr, label='z=0.000')
ptl.xlim(0.02, 2.)
plt.legend()
plt.show()
'''

Traykova_k_arr = FML_GR_L400_Np512_Nmesh1536_z0_k_arr

FML_TraykovaL2_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L11_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")
FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L39_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_screen_forcelink02_smoothR3_cb_z-0.002.txt")

FML_TraykovaL2_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2_GRforce_L400_Np512_Nmesh1536_z0_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt")
FML_TraykovaL2L11_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L11_GRforce_L400_Np512_Nmesh1536_z0_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L11_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt")
FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L39_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z-0.002.txt")

FML_TraykovaL2_unscreen_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2_unscreen_L400_Np512_Nmesh1536_z0_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt")
FML_TraykovaL2L11_unscreen_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L11_unscreen_L400_Np512_Nmesh1536_z0_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L11_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt")
FML_TraykovaL2L39_unscreen_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L39_unscreen_L400_Np512_Nmesh1536_z0_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L39_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_unscreen_cb_z-0.002.txt")

# boost factors
FML_TraykovaL2_GR_ratio_L400_Np512_Nmesh1536_z0_P_arr = [a/b for a,b in zip(FML_TraykovaL2_L400_Np512_Nmesh1536_z0_P_arr, FML_GR_L400_Np512_Nmesh1536_z0_P_arr)]
FML_TraykovaL2L11_GR_ratio_L400_Np512_Nmesh1536_z0_P_arr = [a/b for a,b in zip(FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_P_arr, FML_GR_L400_Np512_Nmesh1536_z0_P_arr)]
FML_TraykovaL2L39_GR_ratio_L400_Np512_Nmesh1536_z0_P_arr = [a/b for a,b in zip(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_P_arr, FML_GR_L400_Np512_Nmesh1536_z0_P_arr)]

FML_TraykovaL2_L400_Np512_Nmesh1536_z0_scr5eff_ratio_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_scr5eff_ratio_P_arr = comp_diff(FML_TraykovaL2_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_P_arr, FML_TraykovaL2_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2_GRforce_L400_Np512_Nmesh1536_z0_P_arr)
FML_TraykovaL2_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_P_arr = comp_diff(FML_TraykovaL2_unscreen_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2_unscreen_L400_Np512_Nmesh1536_z0_P_arr, FML_TraykovaL2_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2_GRforce_L400_Np512_Nmesh1536_z0_P_arr)
FML_TraykovaL2_L400_Np512_Nmesh1536_z0_modbgeff_ratio_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_modbgeff_ratio_P_arr = comp_diff(FML_TraykovaL2_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2_GRforce_L400_Np512_Nmesh1536_z0_P_arr, FML_GR_L400_Np512_Nmesh1536_z0_k_arr, FML_GR_L400_Np512_Nmesh1536_z0_P_arr)
FML_TraykovaL2_L400_Np512_Nmesh1536_z0_screeneff_ratio_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_screeneff_ratio_P_arr = comp_diff(FML_TraykovaL2_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_P_arr, FML_TraykovaL2_unscreen_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2_unscreen_L400_Np512_Nmesh1536_z0_P_arr)

FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_scr5eff_ratio_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_scr5eff_ratio_P_arr = comp_diff(FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_P_arr, FML_TraykovaL2L11_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L11_GRforce_L400_Np512_Nmesh1536_z0_P_arr)
FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_P_arr = comp_diff(FML_TraykovaL2L11_unscreen_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L11_unscreen_L400_Np512_Nmesh1536_z0_P_arr, FML_TraykovaL2L11_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L11_GRforce_L400_Np512_Nmesh1536_z0_P_arr)
FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_modbgeff_ratio_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_modbgeff_ratio_P_arr = comp_diff(FML_TraykovaL2L11_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L11_GRforce_L400_Np512_Nmesh1536_z0_P_arr, FML_GR_L400_Np512_Nmesh1536_z0_k_arr, FML_GR_L400_Np512_Nmesh1536_z0_P_arr)
FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_screeneff_ratio_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_screeneff_ratio_P_arr = comp_diff(FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_P_arr, FML_TraykovaL2L11_unscreen_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L11_unscreen_L400_Np512_Nmesh1536_z0_P_arr)

FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_scr5eff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_scr5eff_ratio_P_arr = comp_diff(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_P_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0_P_arr)
FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_P_arr = comp_diff(FML_TraykovaL2L39_unscreen_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L39_unscreen_L400_Np512_Nmesh1536_z0_P_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0_P_arr)
FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_modbgeff_ratio_P_arr = comp_diff(FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0_P_arr, FML_GR_L400_Np512_Nmesh1536_z0_k_arr, FML_GR_L400_Np512_Nmesh1536_z0_P_arr)
FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_screeneff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_screeneff_ratio_P_arr = comp_diff(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_P_arr, FML_TraykovaL2L39_unscreen_L400_Np512_Nmesh1536_z0_k_arr, FML_TraykovaL2L39_unscreen_L400_Np512_Nmesh1536_z0_P_arr)

# evolution with z
FML_GR_L400_Np512_Nmesh1536_z0p480_k_arr, FML_GR_L400_Np512_Nmesh1536_z0p480_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z0.480.txt")
FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0p480_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0p480_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L39_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z0.480.txt")
FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0p480_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0p480_modbgeff_ratio_P_arr = comp_diff(FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0p480_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0p480_P_arr, FML_GR_L400_Np512_Nmesh1536_z0p480_k_arr, FML_GR_L400_Np512_Nmesh1536_z0p480_P_arr)

FML_GR_L400_Np512_Nmesh1536_z0p985_k_arr, FML_GR_L400_Np512_Nmesh1536_z0p985_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z0.985.txt")
FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0p985_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0p985_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L39_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z0.985.txt")
FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0p985_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0p985_modbgeff_ratio_P_arr = comp_diff(FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0p985_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z0p985_P_arr, FML_GR_L400_Np512_Nmesh1536_z0p985_k_arr, FML_GR_L400_Np512_Nmesh1536_z0p985_P_arr)

FML_GR_L400_Np512_Nmesh1536_z1p990_k_arr, FML_GR_L400_Np512_Nmesh1536_z1p990_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z1.990.txt")
FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z1p990_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z1p990_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L39_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z1.990.txt")
FML_TraykovaL2L39_L400_Np512_Nmesh1536_z1p990_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z1p990_modbgeff_ratio_P_arr = comp_diff(FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z1p990_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z1p990_P_arr, FML_GR_L400_Np512_Nmesh1536_z1p990_k_arr, FML_GR_L400_Np512_Nmesh1536_z1p990_P_arr)

FML_GR_L400_Np512_Nmesh1536_z9p804_k_arr, FML_GR_L400_Np512_Nmesh1536_z9p804_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z9.804.txt")
FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z9p804_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z9p804_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L39_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z9.804.txt")
FML_TraykovaL2L39_L400_Np512_Nmesh1536_z9p804_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z9p804_modbgeff_ratio_P_arr = comp_diff(FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z9p804_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z9p804_P_arr, FML_GR_L400_Np512_Nmesh1536_z9p804_k_arr, FML_GR_L400_Np512_Nmesh1536_z9p804_P_arr)

FML_GR_L400_Np512_Nmesh1536_z13p625_k_arr, FML_GR_L400_Np512_Nmesh1536_z13p625_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z13.625.txt")
FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z13p625_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z13p625_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L39_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z13.625.txt")
FML_TraykovaL2L39_L400_Np512_Nmesh1536_z13p625_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z13p625_modbgeff_ratio_P_arr = comp_diff(FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z13p625_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z13p625_P_arr, FML_GR_L400_Np512_Nmesh1536_z13p625_k_arr, FML_GR_L400_Np512_Nmesh1536_z13p625_P_arr)

FML_GR_L400_Np512_Nmesh1536_z21p631_k_arr, FML_GR_L400_Np512_Nmesh1536_z21p631_P_arr = read_data("output/HiCOLA/pofk_GR_Barreira_L400_Np512_Nmesh1536_HiCOLA_cb_z21.631.txt")
FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z21p631_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z21p631_P_arr = read_data("output/HiCOLA/pofk_TraykovaL2-L39_L400_Np512_Nmesh1536_HiCOLA_modbg_lin2LPT_GRforce_cb_z21.631.txt")
FML_TraykovaL2L39_L400_Np512_Nmesh1536_z21p631_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z21p631_modbgeff_ratio_P_arr = comp_diff(FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z21p631_k_arr, FML_TraykovaL2L39_GRforce_L400_Np512_Nmesh1536_z21p631_P_arr, FML_GR_L400_Np512_Nmesh1536_z21p631_k_arr, FML_GR_L400_Np512_Nmesh1536_z21p631_P_arr)

##################
# Priority Plots #
##################

# scalar field cuGal vs ESS comparison
'''
fig = plt.figure()
plt.loglog(z_arr_inv_f1p0, y_arr_f1p0, '-', color=CB_color_cycle[0], label='cuGal')
plt.loglog(TraykovaL2L39_z_arr_inv, TraykovaL2L39_y_arr, '-.', color=CB_color_cycle[7], label='ESS')
plt.ylabel(r'$y=\tilde{\phi}^{\prime}/\tilde{\phi}^{\prime}_{\rm dS}$')
plt.xlabel(r'$z$', labelpad=0.05)
plt.legend()
fig.set_size_inches(20, 10, forward=True)
plt.tight_layout(pad=0.5)
#plt.show()
'''

# cuGal expansion + P(k) effect
'''
fig, axarr = plt.subplots(nrows=2, ncols=1, sharey=False, sharex=False)
ax1 = axarr[0]
ax2 = axarr[1]
#plt.semilogx(z_arr_inv_f1p0, D_growth_arr_inv_f1p0, 'r-', label='Full')
#plt.semilogx(z_arr_inv_f1p0, D_growth_arr_inv_f1p0_LCDMbg, 'b--', label='LCDMbg')

ax1.axhline(y=1., linestyle=':', color='black', label=r'$f_{\phi}=0$ ($\Lambda$CDM)')
ax1.semilogx(z_arr_inv_f0p2, E_f0p2_modbg_effect, '-', color=CB_color_cycle[7], label=r'$f_{\phi}=0.2, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.2, almost))
ax1.semilogx(z_arr_inv_f0p4, E_f0p4_modbg_effect, '-', color=CB_color_cycle[5], label=r'$f_{\phi}=0.4, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.4, almost))
ax1.semilogx(z_arr_inv_f0p6, E_f0p6_modbg_effect, '-', color=CB_color_cycle[2], label=r'$f_{\phi}=0.6, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.6, almost))
ax1.semilogx(z_arr_inv_f0p8, E_f0p8_modbg_effect, '-', color=CB_color_cycle[1], label=r'$f_{\phi}=0.8, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.8, almost))
ax1.semilogx(z_arr_inv_f1p0, E_f1p0_modbg_effect, '-', color=CB_color_cycle[0], label=r'$f_{\phi}=1.0, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 1.0, almost))
#ax1.set_ylabel(r'$\frac{E^{\rm cuGal}(z)}{E^{\Lambda{\rm CDM}}(z)}$', fontsize=32)
ax1.set_ylabel(r'$E^{\rm cuGal}(z)/E^{\Lambda{\rm CDM}}(z)$')
ax1.set_xlabel(r'$z$', labelpad=0.05)
ax1.legend(fontsize='x-small', loc='right')
ax1.set_xlim(0., 999.)

ax2.axhline(y=1., linestyle=':', color='black')
ax2.semilogx(FML_cG_f1p0_EdSmax_modbgeff_ratio_k_arr, FML_cG_f1p0_EdSmax_modbgeff_ratio_P_arr, '-', color=CB_color_cycle[0], label=r'$z=0.000$')
ax2.semilogx(FML_cG_f1p0_EdSmax_modbgeff_ratio_z0p480_k_arr, FML_cG_f1p0_EdSmax_modbgeff_ratio_z0p480_P_arr, '--', color=CB_color_cycle[0], label=r'$z=0.480$')
ax2.semilogx(FML_cG_f1p0_EdSmax_modbgeff_ratio_z0p985_k_arr, FML_cG_f1p0_EdSmax_modbgeff_ratio_z0p985_P_arr, '-.', color=CB_color_cycle[0], label=r'$z=0.985$')
ax2.semilogx(FML_cG_f1p0_EdSmax_modbgeff_ratio_z1p990_k_arr, FML_cG_f1p0_EdSmax_modbgeff_ratio_z1p990_P_arr, ':', color=CB_color_cycle[0], label=r'$z=1.990$')
#ax2.set_ylabel(r'$\frac{P^{\rm QCDM}(k, z)}{P^{\Lambda{\rm CDM}}(k, z)}$', fontsize=32)
ax2.set_ylabel(r'$P^{\rm QCDM}(k, z)/P^{\Lambda{\rm CDM}}(k, z)$')
ax2.set_xlabel(r'$k\ [h/{\rm Mpc}]$', labelpad=0.05)
ax2.legend(fontsize='x-small')
ax2.set_xlim(0.02, 2.)
ax2.set_ylim(0.99, 1.35)
fig.set_size_inches(20, 10, forward=True)
plt.tight_layout(pad=0.25)
#plt.savefig('Plots/HiCOLA/Paper_plots/final_cuGal_expansion_Pbg.pdf')
#plt.show()
'''

# ESS expansion + P(k) effect

E_TraykovaL2_modbg_effect_interp = interp1d(z_arr_inv_LCDMbg, E_TraykovaL2_modbg_effect, kind='cubic', fill_value="extrapolate")
E_TraykovaL2L11_modbg_effect_interp = interp1d(z_arr_inv_LCDMbg, E_TraykovaL2L11_modbg_effect, kind='cubic', fill_value="extrapolate")
E_TraykovaL2L39_modbg_effect_interp = interp1d(z_arr_inv_LCDMbg, E_TraykovaL2L39_modbg_effect, kind='cubic', fill_value="extrapolate")

tmp_z_arr = np.geomspace(1e-3, np.max(z_arr_inv_LCDMbg), num=60)

E_TraykovaL2_modbg_effect_interp_arr = E_TraykovaL2_modbg_effect_interp(tmp_z_arr)
E_TraykovaL2L11_modbg_effect_interp_arr = E_TraykovaL2L11_modbg_effect_interp(tmp_z_arr)
E_TraykovaL2L39_modbg_effect_interp_arr = E_TraykovaL2L39_modbg_effect_interp(tmp_z_arr)

#plt.rcParams['lines.linewidth'] = 4
fig, axarr = plt.subplots(nrows=2, ncols=1, sharey=False, sharex=False)
ax1 = axarr[0]
ax2 = axarr[1]
ax1.axhline(y=1., linestyle=':', color='black', label=r'$\Lambda$CDM')
#plt.semilogx(z_arr_inv_f1p0, D_growth_arr_inv_f1p0, 'r-', label='Full')
#plt.semilogx(z_arr_inv_f1p0, D_growth_arr_inv_f1p0_LCDMbg, 'b--', label='LCDMbg')
ax1.semilogx(tmp_z_arr, E_TraykovaL2_modbg_effect_interp_arr, '-', color=CB_color_cycle[0], label='ESS-A')#, linewidth=2.0)
ax1.semilogx(tmp_z_arr, E_TraykovaL2L11_modbg_effect_interp_arr, '-', color=CB_color_cycle[2], label='ESS-B')#, linewidth=2.0)
ax1.semilogx(tmp_z_arr, E_TraykovaL2L39_modbg_effect_interp_arr, '-', color=CB_color_cycle[7], label='ESS-C')#, linewidth=2.0)
#ax1.semilogx(z_arr_inv_LCDMbg, E_TraykovaL2_modbg_effect, '-', color=CB_color_cycle[1], linewidth=2.0)#, label='ESS-A bumpy')
#ax1.semilogx(z_arr_inv_LCDMbg, E_TraykovaL2L11_modbg_effect, '-', color=CB_color_cycle[3], linewidth=2.0)#, label='ESS-B bumpy')
#ax1.semilogx(z_arr_inv_LCDMbg, E_TraykovaL2L39_modbg_effect, '-', color=CB_color_cycle[5], linewidth=2.0)#, label='ESS-C bumpy')
#ax1.set_ylabel(r'$\frac{E^{\rm ESS}(z)}{E^{\Lambda{\rm CDM}}(z)}$', fontsize=32)
ax1.set_ylabel(r'$E^{\rm ESS}(z)/E^{\Lambda{\rm CDM}}(z)$')
ax1.set_xlabel(r'$z$', labelpad=0.05)
ax1.legend(fontsize='x-small', loc='upper left')
ax1.set_xlim(0.004, 999.)
ax1.set_ylim(0.998, 1.003)

ax2.axhline(y=1., linestyle=':', color='black')
ax2.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_modbgeff_ratio_P_arr, '-', color=CB_color_cycle[7], label=r'$z=0.000$')
#ax2.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0p480_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0p480_modbgeff_ratio_P_arr, '--', label=r'$z=0.480$')
#ax2.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0p985_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0p985_modbgeff_ratio_P_arr, '-.', label=r'$z=0.985$')
ax2.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z1p990_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z1p990_modbgeff_ratio_P_arr, '--', color=CB_color_cycle[7], label=r'$z=1.990$')
ax2.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z9p804_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z9p804_modbgeff_ratio_P_arr, '-.',  color=CB_color_cycle[7],label=r'$z=9.804$')
ax2.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z21p631_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z21p631_modbgeff_ratio_P_arr, ':', color=CB_color_cycle[7], label=r'$z=21.631$')
#ax2.set_ylabel(r'$\frac{P^{\rm QCDM}(k, z)}{P^{\Lambda{\rm CDM}}(k, z)}$', fontsize=32)
ax2.set_ylabel(r'$P^{\rm QCDM}(k, z)/P^{\Lambda{\rm CDM}}(k, z)$')
ax2.set_xlabel(r'$k\ [h/{\rm Mpc}]$', labelpad=0.05)
ax2.legend(fontsize='x-small')
ax2.set_xlim(0.02, 2.)
ax2.set_ylim(0.9725, 1.001)
fig.set_size_inches(20, 10, forward=True)
plt.tight_layout(pad=0.25)
plt.savefig('Plots/HiCOLA/Paper_plots/final_ESS_expansion_Pbg.pdf', bbox_inches='tight')
plt.show()
#plt.rcParams['lines.linewidth'] = 2


# comparison between cuGal and ESS
'''
fig = plt.figure()
gs = gridspec.GridSpec(3, 2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[:2,1])
ax4 = fig.add_subplot(gs[2,:])

ax1.loglog(z_arr_inv_f1p0, Coupl_arr_f1p0, '-', color=CB_color_cycle[0])
ax1.loglog(TraykovaL2L39_z_arr, TraykovaL2L39_coupl_arr, color=CB_color_cycle[7])
ax1.set_xlim(1e-2, 999.)
ax1.set_ylim(1e-8, 2.)
ax1.set_ylabel(r'$\beta(z)$')
ax1.set_xlabel(r'$z$')

ax2.semilogx(z_arr_inv_f1p0, screen_fac_arr_f1p0_low, '-', color=CB_color_cycle[0])
ax2.semilogx(z_arr_inv_f1p0, screen_fac_arr_f1p0_med, '--', color=CB_color_cycle[0])
ax2.semilogx(z_arr_inv_f1p0, screen_fac_arr_f1p0_high, ':', color=CB_color_cycle[0])
ax2.semilogx(TraykovaL2L39_z_arr, screen_fac_arr_TraykovaL2L39_low, '-', color=CB_color_cycle[7])
ax2.semilogx(TraykovaL2L39_z_arr, screen_fac_arr_TraykovaL2L39_med, '--', color=CB_color_cycle[7])
ax2.semilogx(TraykovaL2L39_z_arr, screen_fac_arr_TraykovaL2L39_high, ':', color=CB_color_cycle[7])
ax2.set_xlim(1e-2, 999.)
ax2.set_ylim(0., 1.05)
ax2.set_ylabel(r'$S(z; \delta_{\rm m})$')
ax2.set_xlabel(r'$z$')

ax3.semilogx(z_arr_inv_f1p0, both_arr_f1p0_low, '-', color=CB_color_cycle[0], label=r'$f_{\phi}=1.0, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 1.0, almost))
ax3.semilogx(z_arr_inv_f1p0, both_arr_f1p0_med, '--', color=CB_color_cycle[0])
ax3.semilogx(z_arr_inv_f1p0, both_arr_f1p0_high, ':', color=CB_color_cycle[0])
ax3.loglog(TraykovaL2L39_z_arr, both_arr_TraykovaL2L39_low, 'k-', label=r'$\delta=%.1e$' % dens_low)
ax3.loglog(TraykovaL2L39_z_arr, both_arr_TraykovaL2L39_med, 'k--', label=r'$\delta=%.1e$' % dens_med)
ax3.loglog(TraykovaL2L39_z_arr, both_arr_TraykovaL2L39_high, 'k:', label=r'$\delta=%.1e$' % dens_high)
ax3.loglog(TraykovaL2L39_z_arr, both_arr_TraykovaL2L39_low, '-', color=CB_color_cycle[7], label='ESS-C')
ax3.loglog(TraykovaL2L39_z_arr, both_arr_TraykovaL2L39_med, '--', color=CB_color_cycle[7])
ax3.loglog(TraykovaL2L39_z_arr, both_arr_TraykovaL2L39_high, ':', color=CB_color_cycle[7])
ax3.set_xlim(1e-2, 999.)
ax3.set_ylim(0., 1.1)
ax3.set_ylabel(r'$\beta\times S$')
ax3.set_xlabel(r'$z$')
ax3.legend(loc='upper right',fontsize=8)

ax4.semilogx(FML_cG_f1p0_EdSmax_scr5eff_ratio_k_arr, FML_cG_f1p0_EdSmax_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[0])
ax4.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_scr5eff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[7])
ax4.set_ylabel(r'$\frac{P^{\rm screened}(k, z=0)}{P^{\rm QCDM}(k, z=0)}$')
ax4.set_xlabel(r'$k\ [h/{\rm Mpc}]$')
ax4.set_xlim(0.02, 2.)
ax4.set_ylim(0.999, 1.11)
plt.show()
'''

###############
# cuGal Plots #
###############

#
# Plots for varing (f, E_dS_max(f))
#
'''
# validity in (f, k_1) space
Npoints = 100
Omega_m_crit = 1.-1e-2

#k1 from E_dS
E_dS_fac_arr = np.linspace(0.8, 0.94, Npoints) #np.linspace(0.9113, 0.9114, Npoints) #np.linspace(0.85, 0.92, Npoints) #np.linspace(0.8, 0.911405, 10) #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]#np.arange(0.1, 0.9+0.1, 0.1) #[0.8, 0.9] #
f_arr = np.linspace(0.0, 1.0, Npoints) #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #np.arange(0., 1.0+0.1, 0.1)
a_arr_inv = np.zeros((len(f_arr),len(E_dS_fac_arr), z_num))
E_arr = np.zeros((len(f_arr),len(E_dS_fac_arr), z_num))
E_prime_E_arr = np.zeros((len(f_arr),len(E_dS_fac_arr), z_num))
Omega_r_arr = np.zeros((len(f_arr),len(E_dS_fac_arr), z_num))
Omega_m_arr = np.zeros((len(f_arr),len(E_dS_fac_arr), z_num))
Omega_L_arr = np.zeros((len(f_arr),len(E_dS_fac_arr), z_num))
Omega_phi_arr = np.zeros((len(f_arr),len(E_dS_fac_arr), z_num))
Omega_phi_arr2 = np.zeros((len(f_arr),len(E_dS_fac_arr), z_num))
w_arr = np.zeros((len(f_arr),len(E_dS_fac_arr), z_num))

fig = plt.figure()
for i_E, Ev in enumerate(E_dS_fac_arr):
    for i_f, fv in enumerate(f_arr):
        Omega_L0 = (1.-fv)*(1.-Omega_r0_test-Omega_m0_test)
        E0 = 1. #by definition
        w0 = 1. #by definition
        y0 = comp_y_cuGal_dS(0.9, Omega_r0_test, Omega_m0_test, Omega_L0, E0, Ev)
        k1 = -6.*y0**2.
        g1 = comp_g1(Omega_r0_test, Omega_m0_test, k1, fv)
        a_arr_inv[i_f, i_E, :], E_arr[i_f, i_E, :], E_prime_E_arr[i_f, i_E, :], Omega_r_arr[i_f, i_E, :], Omega_m_arr[i_f, i_E, :], Omega_L_arr[i_f, i_E, :], Omega_phi_arr[i_f, i_E, :], Omega_phi_arr2[i_f, i_E, :], w_arr[i_f, i_E, :] = run_solver_inv_cuGal_today_bg(z_num, z_max, Omega_r0_test, Omega_m0_test, k1, fv)
        Omega_DE_arr = [OmLv+Ompv for OmLv, Ompv in zip(Omega_L_arr[i_f, i_E, :], Omega_phi_arr[i_f, i_E, :])]
        alpha_fac = 1.-Omega_L_arr[i_f, i_E, 0]/Ev/Ev
        track = -1.*(k1 + 3.*g1*E0*E0*w0) #(E_arr[i_f, i_E, 0]/Ev)**2.*y_arr[i_f, i_E, 0]-1.
        if fv == 0:
            plt.plot(fv, k1, 'g.')
        elif alpha_fac < 0:
            plt.plot(fv, k1, 'y.')
        elif track < 0:
            plt.plot(fv, k1, 'b.')
        elif np.max(Omega_DE_arr) > Omega_DE_arr[0]:
            #print(fv, Ev, 'DE grows')
            plt.plot(fv, k1, 'r.')
        elif np.max(Omega_m_arr[i_f, i_E, :]) < Omega_m_crit:
            #print(fv, Ev, 'Not enough matter')
            plt.plot(fv, k1, 'm.')
        else:
            #print(fv, Ev, 'Fine')
            plt.plot(fv, k1, 'g.')
    #plt.plot(f_arr, Ep_E0_arr[i_E, :], '-', label=r'$E_{\rm dS} = %2f$' %  E_dS_fac_arr[i_E])
crit_EdS_arr = [np.sqrt((1-fvv)*((1.-Omega_r0_test-Omega_m0_test))) for fvv in f_arr]
Omega_L0_arr = [(1.-fvv)*(1.-Omega_r0_test-Omega_m0_test) for fvv in f_arr]
E0 = 1. #by definition
w0 = 1. #by definition
y0_arr = [comp_y_cuGal_dS(0.9, Omega_r0_test, Omega_m0_test, Omega_L0v, E0, Ecv) for Omega_L0v, Ecv in zip(Omega_L0_arr, crit_EdS_arr)]
crit_k1_arr = [-6.*y0v**2. for y0v in y0_arr]

#plt.plot(f_arr, crit_k1_arr, 'k-')
#plt.legend()
plt.xlabel(r'$f_{\phi}$')
plt.ylabel(r'$k_1$')
plt.show()

#direct k1
f_arr = np.linspace(0., 1., Npoints)
k1_arr = np.linspace(-6., 6., Npoints) #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #np.arange(0., 1.0+0.1, 0.1)
a_arr_inv = np.zeros((len(f_arr),len(k1_arr), z_num))
E_arr = np.zeros((len(f_arr),len(k1_arr), z_num))
E_prime_E_arr = np.zeros((len(f_arr),len(k1_arr), z_num))
Omega_r_arr = np.zeros((len(f_arr),len(k1_arr), z_num))
Omega_m_arr = np.zeros((len(f_arr),len(k1_arr), z_num))
Omega_L_arr = np.zeros((len(f_arr),len(k1_arr), z_num))
Omega_phi_arr = np.zeros((len(f_arr),len(k1_arr), z_num))
Omega_phi_arr2 = np.zeros((len(f_arr),len(k1_arr), z_num))
w_arr = np.zeros((len(f_arr),len(k1_arr), z_num))

fig = plt.figure()
for i_k, k1v in enumerate(k1_arr):
    for i_f, fv in enumerate(f_arr):
        Omega_L0 = (1.-fv)*(1.-Omega_r0_test-Omega_m0_test)
        E0 = 1. #by definition
        w0 = 1. #by definition
        g1 = comp_g1(Omega_r0_test, Omega_m0_test, k1v, fv)
        a_arr_inv[i_f, i_k, :], E_arr[i_f, i_k, :], E_prime_E_arr[i_f, i_k, :], Omega_r_arr[i_f, i_k, :], Omega_m_arr[i_f, i_k, :], Omega_L_arr[i_f, i_k, :], Omega_phi_arr[i_f, i_k, :], Omega_phi_arr2[i_f, i_k, :], w_arr[i_f, i_k, :] = run_solver_inv_cuGal_today_bg(z_num, z_max, Omega_r0_test, Omega_m0_test, k1v, fv)
        Omega_DE_arr = [OmLv+Ompv for OmLv, Ompv in zip(Omega_L_arr[i_f, i_k, :], Omega_phi_arr[i_f, i_k, :])]
        #alpha_fac = 1.-Omega_L_arr[i_f, i_k, 0]/Ev/Ev
        track = -1.*(k1v + 3.*g1*E0*E0*w0) #(E_arr[i_f, i_k, 0]/Ev)**2.*y_arr[i_f, i_k, 0]-1.
        if fv == 0:
            plt.plot(fv, k1v, 'g.')
        #elif alpha_fac < 0:
        #    plt.plot(fv, k1v, 'y.')
        elif track < 0:
            plt.plot(fv, k1v, 'b.')
        elif np.max(Omega_DE_arr) > Omega_DE_arr[0]:
            #print(fv, Ev, 'DE grows')
            plt.plot(fv, k1v, 'r.')
        elif np.max(Omega_m_arr[i_f, i_k, :]) < Omega_m_crit:
            #print(fv, Ev, 'Not enough matter')
            plt.plot(fv, k1v, 'm.')
        else:
            #print(fv, Ev, 'Fine')
            plt.plot(fv, k1v, 'g.')
    #plt.plot(f_arr, Ep_E0_arr[i_k, :], '-', label=r'$E_{\rm dS} = %2f$' %  E_dS_fac_arr[i_k])
#plt.plot(f_arr, crit_k1_arr, 'k-')
#plt.legend()
plt.xlabel(r'$f_{\phi}$')
plt.ylabel(r'$k_1$')
plt.show()
'''

# cubic Galileon background

plt.rcParams['lines.linewidth'] = 4#2
fig, axarr = plt.subplots(nrows=3, ncols=2, sharey=False, sharex=True)
ax1 = axarr[0, 0]
ax2 = axarr[0, 1]
ax3 = axarr[1, 0]
ax4 = axarr[1, 1]
ax5 = axarr[2, 0]
ax6 = axarr[2, 1]
#ax1.semilogx(z_arr_inv_f0p0, E_prime_E_arr_f0p0, 'k--', label=r'$f_{\phi}=0.0, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.0, almost))
ax1.semilogx(z_arr_inv_LCDMbg, E_prime_E_arr_LCDMbg, 'k:', label=r'$f_{\phi}=0$, ($\Lambda$CDM)')
ax1.semilogx(z_arr_inv_f0p2, E_prime_E_arr_f0p2, '-', color=CB_color_cycle[7], label=r'$f_{\phi}=0.2, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.2, almost))
ax1.semilogx(z_arr_inv_f0p4, E_prime_E_arr_f0p4, '-', color=CB_color_cycle[5], label=r'$f_{\phi}=0.4, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.4, almost))
ax1.semilogx(z_arr_inv_f0p6, E_prime_E_arr_f0p6, '-', color=CB_color_cycle[2], label=r'$f_{\phi}=0.6, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.6, almost))
ax1.semilogx(z_arr_inv_f0p8, E_prime_E_arr_f0p8, '-', color=CB_color_cycle[1], label=r'$f_{\phi}=0.8, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.8, almost))
ax1.semilogx(z_arr_inv_f1p0, E_prime_E_arr_f1p0, '-', color=CB_color_cycle[0], label=r'$f_{\phi}=1.0, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 1.0, almost))
#ax1.set_yticklabels([-2.0, -1.5, -1.0, -0.5, 0.])
ax1.set_ylabel(r'$E^{\prime}/E$')
ax1.set_ylim(-1.6, 0.)
ax1.legend(loc='upper right', fontsize='15')
ax2.semilogx(z_arr_inv_f0p2, y_arr_f0p2, '-', color=CB_color_cycle[7])#, alpha=0.2)
ax2.semilogx(z_arr_inv_f0p4, y_arr_f0p4, '-', color=CB_color_cycle[5])#, alpha=0.4)
ax2.semilogx(z_arr_inv_f0p6, y_arr_f0p6, '-', color=CB_color_cycle[2])#, alpha=0.6)
ax2.semilogx(z_arr_inv_f0p8, y_arr_f0p8, '-', color=CB_color_cycle[1])#, alpha=0.8)
ax2.semilogx(z_arr_inv_f1p0, y_arr_f1p0, '-', color=CB_color_cycle[0])#, alpha=1.0)
ax2.set_ylabel(r'$y=\tilde{\phi}^{\prime}/\tilde{\phi}^{\prime}_{\rm dS}$')
ax3.semilogx(z_arr_inv_LCDMbg, Omega_r_arr_LCDMbg, 'k:')
ax3.semilogx(z_arr_inv_f0p2, Omega_r_arr_f0p2, '-', color=CB_color_cycle[7])
ax3.semilogx(z_arr_inv_f0p4, Omega_r_arr_f0p4, '-', color=CB_color_cycle[5])
ax3.semilogx(z_arr_inv_f0p6, Omega_r_arr_f0p6, '-', color=CB_color_cycle[2])
ax3.semilogx(z_arr_inv_f0p8, Omega_r_arr_f0p8, '-', color=CB_color_cycle[1])
ax3.semilogx(z_arr_inv_f1p0, Omega_r_arr_f1p0, '-', color=CB_color_cycle[0])
ax3.set_ylabel(r'$\Omega_{\rm r}$')
#ax3.set_ylim(0., 1.)
ax4.semilogx(z_arr_inv_LCDMbg, Omega_m_arr_LCDMbg, 'k:')
ax4.semilogx(z_arr_inv_f0p2, Omega_m_arr_f0p2, '-', color=CB_color_cycle[7])
ax4.semilogx(z_arr_inv_f0p4, Omega_m_arr_f0p4, '-', color=CB_color_cycle[5])
ax4.semilogx(z_arr_inv_f0p6, Omega_m_arr_f0p6, '-', color=CB_color_cycle[2])
ax4.semilogx(z_arr_inv_f0p8, Omega_m_arr_f0p8, '-', color=CB_color_cycle[1])
ax4.semilogx(z_arr_inv_f1p0, Omega_m_arr_f1p0, '-', color=CB_color_cycle[0])
ax4.set_ylabel(r'$\Omega_{\rm m}$')
#ax4.set_ylim(0., 1.)
ax5.semilogx(z_arr_inv_LCDMbg, Omega_L_arr_LCDMbg, 'k:')
ax5.semilogx(z_arr_inv_f0p2, Omega_L_arr_f0p2, '-', color=CB_color_cycle[7])
ax5.semilogx(z_arr_inv_f0p4, Omega_L_arr_f0p4, '-', color=CB_color_cycle[5])
ax5.semilogx(z_arr_inv_f0p6, Omega_L_arr_f0p6, '-', color=CB_color_cycle[2])
ax5.semilogx(z_arr_inv_f0p8, Omega_L_arr_f0p8, '-', color=CB_color_cycle[1])
ax5.semilogx(z_arr_inv_f1p0, Omega_L_arr_f1p0, '-', color=CB_color_cycle[0])
ax5.set_xlabel(r'$z$', labelpad=0.05)
ax5.set_ylabel(r'$\Omega_{\Lambda}$')
#ax5.set_ylim(0., 1.)
ax6.semilogx(z_arr_inv_f0p2, Omega_phi_arr2_f0p2, '-', color=CB_color_cycle[7])
ax6.semilogx(z_arr_inv_f0p4, Omega_phi_arr2_f0p4, '-', color=CB_color_cycle[5])
ax6.semilogx(z_arr_inv_f0p6, Omega_phi_arr2_f0p6, '-', color=CB_color_cycle[2])
ax6.semilogx(z_arr_inv_f0p8, Omega_phi_arr2_f0p8, '-', color=CB_color_cycle[1])
ax6.semilogx(z_arr_inv_f1p0, Omega_phi_arr2_f1p0, '-', color=CB_color_cycle[0])
ax6.set_xlabel(r'$z$', labelpad=0.05)
ax6.set_ylabel(r'$\Omega_{\phi}$')
#ax6.set_ylim(0., 1.)
fig.set_size_inches(20, 10, forward=True)
plt.tight_layout(pad=0.5)
#plt.savefig('Plots/HiCOLA/Paper_plots/final_cuGal_background.pdf', bbox_inches='tight')
#plt.show()
plt.rcParams['lines.linewidth'] = 4


# cuGal coupling and screening factor

fig = plt.figure()
gs1 = gridspec.GridSpec(2, 2, left=0.097, right=0.985, bottom=0.475, top=0.985, hspace=0.08, wspace=0.25)
ax1 = fig.add_subplot(gs1[0,0])
ax2 = fig.add_subplot(gs1[1,0])
ax3 = fig.add_subplot(gs1[:,1])
gs2 = gridspec.GridSpec(1, 1, left=0.097, right=0.985, bottom=0.0975, top=0.375)
ax4 = fig.add_subplot(gs2[:,:])

ax1.loglog(z_arr_inv_f0p2, Coupl_arr_f0p2, '-', color=CB_color_cycle[7])
#ax1.loglog(z_arr_inv_f0p4, Coupl_arr_f0p4, '-', color=CB_color_cycle[5])
ax1.loglog(z_arr_inv_f0p6, Coupl_arr_f0p6, '-', color=CB_color_cycle[2])
#ax1.loglog(z_arr_inv_f0p8, Coupl_arr_f0p8, '-', color=CB_color_cycle[1])
ax1.loglog(z_arr_inv_f1p0, Coupl_arr_f1p0, '-', color=CB_color_cycle[0])

ax1.set_ylabel(r'$\beta(z)$')
#ax1.set_xlabel(r'$z$')
ax1.set_xticklabels([])
ax1.set_xlim(1e-2, 999.)

ax2.semilogx(z_arr_inv_f0p2, screen_fac_arr_f0p2_low, '-', color=CB_color_cycle[7])
#ax2.semilogx(z_arr_inv_f0p4, screen_fac_arr_f0p4_low, '-', color=CB_color_cycle[5])
ax2.semilogx(z_arr_inv_f0p6, screen_fac_arr_f0p6_low, '-', color=CB_color_cycle[2])
#ax2.semilogx(z_arr_inv_f0p8, screen_fac_arr_f0p8_low, '-', color=CB_color_cycle[1])
ax2.semilogx(z_arr_inv_f1p0, screen_fac_arr_f1p0_low, '-', color=CB_color_cycle[0])

ax2.semilogx(z_arr_inv_f0p2, screen_fac_arr_f0p2_med, '--', color=CB_color_cycle[7])
#ax2.semilogx(z_arr_inv_f0p4, screen_fac_arr_f0p4_med, '--', color=CB_color_cycle[5])
ax2.semilogx(z_arr_inv_f0p6, screen_fac_arr_f0p6_med, '--', color=CB_color_cycle[2])
#ax2.semilogx(z_arr_inv_f0p8, screen_fac_arr_f0p8_med, '--', color=CB_color_cycle[1])
ax2.semilogx(z_arr_inv_f1p0, screen_fac_arr_f1p0_med, '--', color=CB_color_cycle[0])

ax2.semilogx(z_arr_inv_f0p2, screen_fac_arr_f0p2_high, ':', color=CB_color_cycle[7])
#ax2.semilogx(z_arr_inv_f0p4, screen_fac_arr_f0p4_high, ':', color=CB_color_cycle[5])
ax2.semilogx(z_arr_inv_f0p6, screen_fac_arr_f0p6_high, ':', color=CB_color_cycle[2])
#ax2.semilogx(z_arr_inv_f0p8, screen_fac_arr_f0p8_high, ':', color=CB_color_cycle[1])
ax2.semilogx(z_arr_inv_f1p0, screen_fac_arr_f1p0_high, ':', color=CB_color_cycle[0])

ax2.set_ylabel(r'$S(z; \delta_{\rm m})$')
ax2.set_xlabel(r'$z$', labelpad=0.05)
ax2.set_xlim(1e-2, 999.)
ax2.set_ylim(-0.01, 1.05)
#ax2.set_ylim(1e-2, 1.05)

ax3.loglog(z_arr_inv_f1p0, both_arr_f1p0_low, 'k-', label=r'$\delta_{\rm m}=10^{-3}$')#label=r'$\delta=%.1e$' % dens_low)
ax3.loglog(z_arr_inv_f1p0, both_arr_f1p0_med, 'k--', label=r'$\delta_{\rm m}=1$')#label=r'$\delta=%.1e$' % dens_med)
ax3.loglog(z_arr_inv_f1p0, both_arr_f1p0_high, 'k:', label=r'$\delta_{\rm m}=10^{3}$')#label=r'$\delta=%.1e$' % dens_high)

ax3.loglog(z_arr_inv_f0p2, both_arr_f0p2_low, '-', color=CB_color_cycle[7], label=r'$f_{\phi}=0.2, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.2, almost))
#ax3.loglog(z_arr_inv_f0p4, both_arr_f0p4_low, '-', color=CB_color_cycle[5], label=r'$f_{\phi}=0.4, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.4, almost))
ax3.loglog(z_arr_inv_f0p6, both_arr_f0p6_low, '-', color=CB_color_cycle[2], label=r'$f_{\phi}=0.6, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.6, almost))
#ax3.loglog(z_arr_inv_f0p8, both_arr_f0p8_low, '-', color=CB_color_cycle[1], label=r'$f_{\phi}=0.8, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.8, almost))
ax3.loglog(z_arr_inv_f1p0, both_arr_f1p0_low, '-', color=CB_color_cycle[0], label=r'$f_{\phi}=1.0, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 1.0, almost))

ax3.loglog(z_arr_inv_f0p2, both_arr_f0p2_med, '--', color=CB_color_cycle[7])
#ax3.loglog(z_arr_inv_f0p4, both_arr_f0p4_med, '--', color=CB_color_cycle[5])
ax3.loglog(z_arr_inv_f0p6, both_arr_f0p6_med, '--', color=CB_color_cycle[2])
#ax3.loglog(z_arr_inv_f0p8, both_arr_f0p8_med, '--', color=CB_color_cycle[1])
ax3.loglog(z_arr_inv_f1p0, both_arr_f1p0_med, '--', color=CB_color_cycle[0])

ax3.loglog(z_arr_inv_f0p2, both_arr_f0p2_high, ':', color=CB_color_cycle[7])
#ax3.loglog(z_arr_inv_f0p4, both_arr_f0p4_high, ':', color=CB_color_cycle[5])
ax3.loglog(z_arr_inv_f0p6, both_arr_f0p6_high, ':', color=CB_color_cycle[2])
#ax3.loglog(z_arr_inv_f0p8, both_arr_f0p8_high, ':', color=CB_color_cycle[1])
ax3.loglog(z_arr_inv_f1p0, both_arr_f1p0_high, ':', color=CB_color_cycle[0])
ax3.set_ylabel(r'$\beta\times S$')
ax3.set_xlabel(r'$z$', labelpad=0.05)
ax3.legend(fontsize=18, loc='lower left')
#ax3.set_ylim(-0.01, 1.01)
ax3.set_xlim(1e-2, 999.)

ax4.semilogx(FML_cG_f0p2_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p2_EdSmax_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[7])
#ax4.semilogx(FML_cG_f0p4_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p4_EdSmax_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[5])
ax4.semilogx(FML_cG_f0p6_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p6_EdSmax_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[2])
#ax4.semilogx(FML_cG_f0p8_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p8_EdSmax_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[1])
ax4.semilogx(FML_cG_f1p0_EdSmax_scr5eff_ratio_k_arr, FML_cG_f1p0_EdSmax_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[0])
ax4.set_ylabel(r'$\frac{P^{\rm cuGal}(k, z=0)}{P^{\rm QCDM}(k, z=0)}$', fontsize='x-large')
#ax4.set_ylim(0.999, 1.075)
ax4.set_xlabel(r'$k\ [h/{\rm Mpc}]$', labelpad=0.05)
ax4.set_xlim(0.02, 2.)
fig.set_size_inches(20, 10, forward=True)
#plt.tight_layout(pad=1)
#plt.savefig('Plots/HiCOLA/Paper_plots/final_cuGal_coupl_screencoeff_Pk5th.pdf', bbox_inches='tight')
#plt.show()


# Linear growth

fig, axarr = plt.subplots(nrows=2, ncols=1, sharey=False, sharex=True)
ax1 = axarr[0]
ax2 = axarr[1]
ax1.axhline(y=1., linestyle=':', color='black')
#plt.semilogx(z_arr_inv_f1p0, D_growth_arr_inv_f1p0, 'c-', label='Full')
#plt.semilogx(z_arr_inv_f1p0, D_growth_arr_inv_f1p0_LCDMbg, 'b--', label='LCDMbg')

ax1.semilogx(z_arr_inv_f1p0, E_f1p0_modbg_effect, '-', color=CB_color_cycle[7], label=r'$E$')
ax1.semilogx(z_arr_inv_f1p0, E_prime_E_f1p0_modbg_effect, '-', color=CB_color_cycle[0], label=r'$E^{\prime}/E$')
#ax1.semilogx(z_arr_inv_f1p0, Omega_r_f1p0_modbg_effect, '-', label=r'$\Omega_{\rm r}$')
#ax1.semilogx(z_arr_inv_f1p0, Omega_m_f1p0_modbg_effect, 'b--', label=r'$\Omega_{\rm m}$')
#ax1.semilogx(z_arr_inv_f1p0, Omega_L_f1p0_modbg_effect, ':', label=r'$\Omega_{\Lambda}$')
ax1.semilogx(z_arr_inv_f1p0, GeffOverG_f1p0_effect, '-', color=CB_color_cycle[2], label=r'$G_{\rm eff}/G_{\rm N}$')

ax2.axhline(y=1., linestyle=':', color='black')
#ax2.semilogx(z_arr_inv_f1p0, D_growth_arr_inv_f1p0_Omm_effect, 'r-.', label=r'$D_E$')
#ax2.semilogx(z_arr_inv_f1p0, D_growth_arr_inv_f1p0_EpE_effect, 'b-.', label=r'$D_{E^{\prime}/E}$')
ax2.semilogx(z_arr_inv_f1p0, D_growth_arr_inv_f1p0_modbg_effect, '--', color=CB_color_cycle[5], label=r'$D_{\rm QCDM}$', zorder=2)#=D_E+D_{E^{\prime}/E}$')
ax2.semilogx(z_arr_inv_f1p0, D_growth_arr_inv_f1p0_Geff_effect, '--', color=CB_color_cycle[2], label=r'$D_{G_{\rm eff}}$', zorder=3)
ax2.semilogx(z_arr_inv_f1p0, D_growth_arr_inv_f1p0_full_effect, '-', color=CB_color_cycle[1], label=r'$D_{\rm cG}=D_{\rm QCDM}+D_{G_{\rm eff}}$', zorder=1)
ax1.set_ylabel(r'$X^{\rm cuGal}(z)/X^{\Lambda{\rm CDM}}(z)$')
ax2.set_ylabel(r'$D_1^{\rm cuGal}(z)/D_1^{\Lambda{\rm CDM}}(z)$')
ax2.set_xlabel(r'$z$', labelpad=0.05)
ax1.legend(fontsize='small')
ax2.legend(fontsize='small')
ax1.set_xlim(0., 10.)
fig.set_size_inches(20, 10, forward=True)
plt.tight_layout(pad=0.1)
#plt.savefig('Plots/HiCOLA/Paper_plots/final_cuGal_linear_growth.pdf', bbox_inches='tight')
#plt.show()


# Overall P(k) effect

fig = plt.figure()
#plt.semilogx(FML_cG_f1p0_EdSmax_GR_ratio_k_arr, FML_cG_f1p0_EdSmax_GR_ratio_P_arr, 'k-', label='Full mod')
#plt.semilogx(FML_cG_f1p0_EdSmax_LCDMbg_GR_ratio_k_arr, FML_cG_f1p0_EdSmax_LCDMbg_GR_ratio_P_arr, 'k--', label='LCDM bg')
#plt.semilogx(FML_cG_f1p0_EdSmax_unscreen_GR_ratio_k_arr, FML_cG_f1p0_EdSmax_unscreen_GR_ratio_P_arr, 'k-.', label='unscreen')
#plt.semilogx(FML_cG_f1p0_EdSmax_GRforce_GR_ratio_k_arr, FML_cG_f1p0_EdSmax_GRforce_GR_ratio_P_arr, 'k:', label='GR force')

plt.axhline(y=1., color='k', linestyle=':', label=r'$f_{\phi}=0$ ($\Lambda$CDM)')
plt.semilogx(FML_cG_f0p2_EdSmax_GR_ratio_k_arr, FML_cG_f0p2_EdSmax_GR_ratio_P_arr, '-', color=CB_color_cycle[7], label=r'$f_{\phi}=0.2, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.2, almost))
plt.semilogx(FML_cG_f0p4_EdSmax_GR_ratio_k_arr, FML_cG_f0p4_EdSmax_GR_ratio_P_arr, '-', color=CB_color_cycle[5], label=r'$f_{\phi}=0.4, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.4, almost))
plt.semilogx(FML_cG_f0p6_EdSmax_GR_ratio_k_arr, FML_cG_f0p6_EdSmax_GR_ratio_P_arr, '-', color=CB_color_cycle[2], label=r'$f_{\phi}=0.6, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.6, almost))
plt.semilogx(FML_cG_f0p8_EdSmax_GR_ratio_k_arr, FML_cG_f0p8_EdSmax_GR_ratio_P_arr, '-', color=CB_color_cycle[1], label=r'$f_{\phi}=0.8, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.8, almost))
plt.semilogx(FML_cG_f1p0_EdSmax_GR_ratio_k_arr, FML_cG_f1p0_EdSmax_GR_ratio_P_arr, '-', color=CB_color_cycle[0], label=r'$f_{\phi}=1.0, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 1.0, almost))

plt.xlim(0.02, 2.)
plt.ylim(0.99, 1.36)#1.325)
plt.xlabel(r'$k\ [h/{\rm Mpc}]$', labelpad=0.05)
plt.ylabel(r'$P^{\rm cuGal}(k, z=0)/P^{\Lambda{\rm CDM}}(k, z=0)$')
plt.legend(fontsize='x-small')
fig.set_size_inches(20, 10, forward=True)
plt.tight_layout(pad=0.1)
#plt.savefig('Plots/HiCOLA/Paper_plots/final_cuGal_Pk_GR_ratio.pdf', bbox_inches='tight')
#plt.show()


# P(k) effects breakdown

fig, axarr = plt.subplots(nrows=2, ncols=2, sharey=False, sharex=True)
ax1 = axarr[0,0]
ax2 = axarr[0,1]
ax3 = axarr[1,0]
ax4 = axarr[1,1]

ax1.semilogx(FML_cG_f0p2_EdSmax_modbgeff_ratio_k_arr, FML_cG_f0p2_EdSmax_modbgeff_ratio_P_arr, '-', color=CB_color_cycle[7])
ax1.semilogx(FML_cG_f0p4_EdSmax_modbgeff_ratio_k_arr, FML_cG_f0p4_EdSmax_modbgeff_ratio_P_arr, '-', color=CB_color_cycle[5])
ax1.semilogx(FML_cG_f0p6_EdSmax_modbgeff_ratio_k_arr, FML_cG_f0p6_EdSmax_modbgeff_ratio_P_arr, '-', color=CB_color_cycle[2])
ax1.semilogx(FML_cG_f0p8_EdSmax_modbgeff_ratio_k_arr, FML_cG_f0p8_EdSmax_modbgeff_ratio_P_arr, '-', color=CB_color_cycle[1])
ax1.semilogx(FML_cG_f1p0_EdSmax_modbgeff_ratio_k_arr, FML_cG_f1p0_EdSmax_modbgeff_ratio_P_arr, '-', color=CB_color_cycle[0])
ax1.set_ylabel(r'$\frac{P^{\rm QCDM}(k, z=0)}{P^{\Lambda{\rm CDM}}(k, z=0)}$', fontsize='x-large')
ax1.set_ylim(0.99, 1.25)
ax1.set_xlim(0.02, 2.)

ax2.semilogx(FML_cG_f0p2_EdSmax_unscr5eff_ratio_k_arr, FML_cG_f0p2_EdSmax_unscr5eff_ratio_P_arr, '-', color=CB_color_cycle[7], label=r'$f_{\phi}=0.2, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.2, almost))
ax2.semilogx(FML_cG_f0p4_EdSmax_unscr5eff_ratio_k_arr, FML_cG_f0p4_EdSmax_unscr5eff_ratio_P_arr, '-', color=CB_color_cycle[5], label=r'$f_{\phi}=0.4, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.4, almost))
ax2.semilogx(FML_cG_f0p6_EdSmax_unscr5eff_ratio_k_arr, FML_cG_f0p6_EdSmax_unscr5eff_ratio_P_arr, '-', color=CB_color_cycle[2], label=r'$f_{\phi}=0.6, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.6, almost))
ax2.semilogx(FML_cG_f0p8_EdSmax_unscr5eff_ratio_k_arr, FML_cG_f0p8_EdSmax_unscr5eff_ratio_P_arr, '-', color=CB_color_cycle[1], label=r'$f_{\phi}=0.8, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 0.8, almost))
ax2.semilogx(FML_cG_f1p0_EdSmax_unscr5eff_ratio_k_arr, FML_cG_f1p0_EdSmax_unscr5eff_ratio_P_arr, '-', color=CB_color_cycle[0], label=r'$f_{\phi}=1.0, E_{\rm dS}=%.3f$' % comp_E_dS_max(E_dS_max_guess, Omega_r0_test, Omega_m0_test, 1.0, almost))
ax2.set_ylabel(r'$\frac{P^{\rm lin\, cuGal}(k, z=0)}{P^{\rm QCDM}(k, z=0)}$', fontsize='x-large')
ax2.set_ylim(0.99, 1.8)
ax2.legend(fontsize='small')

ax3.semilogx(FML_cG_f0p2_EdSmax_screeneff_ratio_k_arr, FML_cG_f0p2_EdSmax_screeneff_ratio_P_arr, '-', color=CB_color_cycle[7])
ax3.semilogx(FML_cG_f0p4_EdSmax_screeneff_ratio_k_arr, FML_cG_f0p4_EdSmax_screeneff_ratio_P_arr, '-', color=CB_color_cycle[5])
ax3.semilogx(FML_cG_f0p6_EdSmax_screeneff_ratio_k_arr, FML_cG_f0p6_EdSmax_screeneff_ratio_P_arr, '-', color=CB_color_cycle[2])
ax3.semilogx(FML_cG_f0p8_EdSmax_screeneff_ratio_k_arr, FML_cG_f0p8_EdSmax_screeneff_ratio_P_arr, '-', color=CB_color_cycle[1])
ax3.semilogx(FML_cG_f1p0_EdSmax_screeneff_ratio_k_arr, FML_cG_f1p0_EdSmax_screeneff_ratio_P_arr, '-', color=CB_color_cycle[0])
ax3.set_ylabel(r'$\frac{P^{\rm cuGal}(k, z=0)}{P^{\rm lin\, cuGal}(k, z=0)}$', fontsize='x-large')
ax3.set_ylim(0.55, 1.01)

ax4.semilogx(FML_cG_f0p2_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p2_EdSmax_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[7])
ax4.semilogx(FML_cG_f0p4_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p4_EdSmax_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[5])
ax4.semilogx(FML_cG_f0p6_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p6_EdSmax_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[2])
ax4.semilogx(FML_cG_f0p8_EdSmax_scr5eff_ratio_k_arr, FML_cG_f0p8_EdSmax_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[1])
ax4.semilogx(FML_cG_f1p0_EdSmax_scr5eff_ratio_k_arr, FML_cG_f1p0_EdSmax_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[0])
ax4.set_ylabel(r'$\frac{P^{\rm cuGal}(k, z=0)}{P^{\rm QCDM}(k, z=0)}$', fontsize='x-large')
ax4.set_ylim(0.999, 1.075)

ax3.set_xlabel(r'$k\ [h/{\rm Mpc}]$', labelpad=0.05)
ax4.set_xlabel(r'$k\ [h/{\rm Mpc}]$', labelpad=0.05)
fig.set_size_inches(20, 10, forward=True)
plt.tight_layout(pad=0.6)
#plt.savefig('Plots/HiCOLA/Paper_plots/final_cuGal_Pk_effects_breakdown.pdf', bbox_inches='tight')
#plt.show()


##################
# Traykova Plots #
##################

#Traykova background
plt.rcParams['lines.linewidth'] = 4#2
fig, axarr = plt.subplots(nrows=3, ncols=2, sharey=False, sharex=True)
ax1 = axarr[0, 0]
ax2 = axarr[0, 1]
ax3 = axarr[1, 0]
ax4 = axarr[1, 1]
ax5 = axarr[2, 0]
ax6 = axarr[2, 1]
ax1.semilogx(z_arr_inv_LCDMbg, E_prime_E_arr_LCDMbg, 'k:', label=r'$\Lambda$CDM')
ax1.semilogx(TraykovaL2_z_arr_inv, TraykovaL2_E_prime_E_arr, '-', color=CB_color_cycle[0], label='ESS-A')
ax1.semilogx(TraykovaL2L11_z_arr_inv, TraykovaL2L11_E_prime_E_arr, '--', color=CB_color_cycle[2], label='ESS-B')
ax1.semilogx(TraykovaL2L39_z_arr_inv, TraykovaL2L39_E_prime_E_arr, '-.', color=CB_color_cycle[7], label='ESS-C')
ax1.set_ylabel(r'$E^{\prime}/E$')
ax1.set_ylim(-1.6, 0.)
ax1.legend(fontsize='small', loc='upper right')
ax2.semilogx(TraykovaL2_z_arr_inv, TraykovaL2_y_arr, '-', color=CB_color_cycle[0])
ax2.semilogx(TraykovaL2L11_z_arr_inv, TraykovaL2L11_y_arr, '--', color=CB_color_cycle[2])
ax2.semilogx(TraykovaL2L39_z_arr_inv, TraykovaL2L39_y_arr, '-.', color=CB_color_cycle[7])
ax2.set_ylabel(r'$y=\tilde{\phi}^{\prime}/\tilde{\phi}^{\prime}_{\rm dS}$')
ax3.semilogx(TraykovaL2_z_arr_inv, TraykovaL2_Omega_r_arr, '-', color=CB_color_cycle[0])
ax3.semilogx(TraykovaL2L11_z_arr_inv, TraykovaL2L11_Omega_r_arr, '--', color=CB_color_cycle[2])
ax3.semilogx(TraykovaL2L39_z_arr_inv, TraykovaL2L39_Omega_r_arr, '-.', color=CB_color_cycle[7])
ax3.semilogx(z_arr_inv_LCDMbg, Omega_r_arr_LCDMbg, 'k:')
ax3.set_ylabel(r'$\Omega_{\rm r}$')
ax3.set_ylim(-0.001, 0.03)
ax4.semilogx(TraykovaL2_z_arr_inv, TraykovaL2_Omega_m_arr, '-', color=CB_color_cycle[0])
ax4.semilogx(TraykovaL2L11_z_arr_inv, TraykovaL2L11_Omega_m_arr, '--', color=CB_color_cycle[2])
ax4.semilogx(TraykovaL2L39_z_arr_inv, TraykovaL2L39_Omega_m_arr, '-.', color=CB_color_cycle[7])
#ax4.semilogx(TraykovaL2_z_arr_inv, 0.99*np.ones((len(TraykovaL2_z_arr_inv),)), 'k:')
#ax4.semilogx(TraykovaL2_z_arr_inv, 0.999*np.ones((len(TraykovaL2_z_arr_inv),)), 'k:')
ax4.semilogx(z_arr_inv_LCDMbg, Omega_m_arr_LCDMbg, 'k:')
ax4.set_ylabel(r'$\Omega_{\rm m}$')
#ax4.set_ylim(0., 1.)
ax5.semilogx(TraykovaL2_z_arr_inv, TraykovaL2_Omega_L_arr, '-', color=CB_color_cycle[0])
ax5.semilogx(TraykovaL2L11_z_arr_inv, TraykovaL2L11_Omega_L_arr, '--', color=CB_color_cycle[2])
ax5.semilogx(TraykovaL2L39_z_arr_inv, TraykovaL2L39_Omega_L_arr, '-.', color=CB_color_cycle[7])
ax5.semilogx(z_arr_inv_LCDMbg, Omega_L_arr_LCDMbg, 'k:')
ax5.set_xlabel(r'$z$', labelpad=0.05)
ax5.set_ylabel(r'$\Omega_{\Lambda}$')
#ax5.set_ylim(0., 1.)
ax6.semilogx(TraykovaL2_z_arr_inv, TraykovaL2_Omega_phi_arr, '-', color=CB_color_cycle[0])
ax6.semilogx(TraykovaL2L11_z_arr_inv, TraykovaL2L11_Omega_phi_arr, '--', color=CB_color_cycle[2])
ax6.semilogx(TraykovaL2L39_z_arr_inv, TraykovaL2L39_Omega_phi_arr, '-.', color=CB_color_cycle[7])
ax6.set_xlabel(r'$z$', labelpad=0.05)
ax6.set_ylabel(r'$\Omega_{\phi}$')
#ax6.set_ylim(0., 1.)
ax6.set_xlim(0., 100.)
fig.set_size_inches(20, 10, forward=True)
plt.tight_layout(pad=0.5)
#plt.savefig('Plots/HiCOLA/Paper_plots/final_ESS_background.pdf', bbox_inches='tight')
#plt.show()
plt.rcParams['lines.linewidth'] = 4


# plot of couplings and screening factor

fig = plt.figure()
gs1 = gridspec.GridSpec(2, 2, left=0.097, right=0.985, bottom=0.475, top=0.985, hspace=0.08, wspace=0.25)
ax1 = fig.add_subplot(gs1[0,0])
ax2 = fig.add_subplot(gs1[1,0])
ax3 = fig.add_subplot(gs1[:,1])
gs2 = gridspec.GridSpec(1, 1, left=0.097, right=0.985, bottom=0.12, top=0.375)
ax4 = fig.add_subplot(gs2[:,:])

ax1.loglog(TraykovaL2_z_arr, TraykovaL2_coupl_arr, color=CB_color_cycle[0])
ax1.loglog(TraykovaL2L11_z_arr, TraykovaL2L11_coupl_arr, color=CB_color_cycle[2])
ax1.loglog(TraykovaL2L39_z_arr, TraykovaL2L39_coupl_arr, color=CB_color_cycle[7])
ax1.set_xlim(1e-2, 999.)
#ax1.set_ylim(1e-4, 2.)
ax1.set_ylabel(r'$\beta(z)$')
#ax1.set_xlabel(r'$z$')
ax1.set_xticklabels([])
ax2.semilogx(TraykovaL2_z_arr, screen_fac_arr_TraykovaL2_low, '-', color=CB_color_cycle[0])
ax2.semilogx(TraykovaL2_z_arr, screen_fac_arr_TraykovaL2_med, '--', color=CB_color_cycle[0])
ax2.semilogx(TraykovaL2_z_arr, screen_fac_arr_TraykovaL2_high, ':', color=CB_color_cycle[0])
ax2.semilogx(TraykovaL2L11_z_arr, screen_fac_arr_TraykovaL2L11_low, '-', color=CB_color_cycle[2])
ax2.semilogx(TraykovaL2L11_z_arr, screen_fac_arr_TraykovaL2L11_med, '--', color=CB_color_cycle[2])
ax2.semilogx(TraykovaL2L11_z_arr, screen_fac_arr_TraykovaL2L11_high, ':', color=CB_color_cycle[2])
ax2.semilogx(TraykovaL2L39_z_arr, screen_fac_arr_TraykovaL2L39_low, '-', color=CB_color_cycle[7])
ax2.semilogx(TraykovaL2L39_z_arr, screen_fac_arr_TraykovaL2L39_med, '--', color=CB_color_cycle[7])
ax2.semilogx(TraykovaL2L39_z_arr, screen_fac_arr_TraykovaL2L39_high, ':', color=CB_color_cycle[7])
ax2.set_xlim(1e-2, 999.)
ax2.set_ylim(0., 1.05)
ax2.set_ylabel(r'$S(z; \delta_{\rm m})$')
ax2.set_xlabel(r'$z$', labelpad=0.05)
ax3.loglog(TraykovaL2_z_arr, both_arr_TraykovaL2_low, 'k-', label=r'$\delta_{\rm m}=10^{-3}$')#label=r'$\delta=%.1e$' % dens_low)
ax3.loglog(TraykovaL2_z_arr, both_arr_TraykovaL2_med, 'k--', label=r'$\delta_{\rm m}=1$')#label=r'$\delta=%.1e$' % dens_med)
ax3.loglog(TraykovaL2_z_arr, both_arr_TraykovaL2_high, 'k:', label=r'$\delta_{\rm m}=10^{3}$')#label=r'$\delta=%.1e$' % dens_high)
ax3.loglog(TraykovaL2_z_arr, both_arr_TraykovaL2_low, '-', color=CB_color_cycle[0], label='ESS-A')
ax3.loglog(TraykovaL2_z_arr, both_arr_TraykovaL2_med, '--', color=CB_color_cycle[0])
ax3.loglog(TraykovaL2_z_arr, both_arr_TraykovaL2_high, ':', color=CB_color_cycle[0])
ax3.loglog(TraykovaL2L11_z_arr, both_arr_TraykovaL2L11_low, '-', color=CB_color_cycle[2], label='ESS-B')
ax3.loglog(TraykovaL2L11_z_arr, both_arr_TraykovaL2L11_med, '--', color=CB_color_cycle[2])
ax3.loglog(TraykovaL2L11_z_arr, both_arr_TraykovaL2L11_high, ':', color=CB_color_cycle[2])
ax3.loglog(TraykovaL2L39_z_arr, both_arr_TraykovaL2L39_low, '-', color=CB_color_cycle[7], label='ESS-C')
ax3.loglog(TraykovaL2L39_z_arr, both_arr_TraykovaL2L39_med, '--', color=CB_color_cycle[7])
ax3.loglog(TraykovaL2L39_z_arr, both_arr_TraykovaL2L39_high, ':', color=CB_color_cycle[7])
ax3.set_xlim(1e-2, 999.)
ax3.set_ylim(0., 1.1)
ax3.set_ylabel(r'$\beta\times S$')
ax3.set_xlabel(r'$z$', labelpad=0.05)
ax3.legend(loc='upper right', fontsize='x-small')#,fontsize=8)

ax4.semilogx(FML_TraykovaL2_L400_Np512_Nmesh1536_z0_scr5eff_ratio_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[0])
ax4.semilogx(FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_scr5eff_ratio_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[2])
ax4.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_scr5eff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[7])
ax4.set_ylabel(r'$\frac{P^{\rm ESS}(k, z=0)}{P^{\rm QCDM}(k, z=0)}$', fontsize='x-large')
ax4.set_xlabel(r'$k\ [h/{\rm Mpc}]$')
ax4.set_xlim(0.02, 2.)
ax4.set_ylim(0.999, 1.11)
fig.set_size_inches(20, 10, forward=True)
#plt.tight_layout(pad=1)
#plt.savefig('Plots/HiCOLA/Paper_plots/final_ESS_coupl_screencoeff_Pk5th.pdf', bbox_inches='tight')
#plt.show()


# Linear growth

fig, axarr = plt.subplots(nrows=2, ncols=1, sharey=False, sharex=True)
ax1 = axarr[0]
ax2 = axarr[1]
ax1.axhline(y=1., linestyle=':', color='black')
#plt.semilogx(z_arr_inv_LCDMbg, D_growth_arr_inv_TraykovaL2L39, 'r-', label='Full')
#plt.semilogx(z_arr_inv_LCDMbg, D_growth_arr_inv_TraykovaL2L39_LCDMbg, 'b--', label='LCDMbg')

ax1.semilogx(z_arr_inv_LCDMbg, E_TraykovaL2L39_modbg_effect, '-', color=CB_color_cycle[7], label=r'$E$')
ax1.semilogx(z_arr_inv_LCDMbg, E_prime_E_TraykovaL2L39_modbg_effect, '-', color=CB_color_cycle[0], label=r'$E^{\prime}/E$')
#ax1.semilogx(z_arr_inv_LCDMbg, Omega_r_TraykovaL2L39_modbg_effect, '-', label=r'$\Omega_{\rm r}$')
#ax1.semilogx(z_arr_inv_LCDMbg, Omega_m_TraykovaL2L39_modbg_effect, 'b--', label=r'$\Omega_{\rm m}$')
#ax1.semilogx(z_arr_inv_LCDMbg, Omega_L_TraykovaL2L39_modbg_effect, ':', label=r'$\Omega_{\Lambda}$')
ax1.semilogx(z_arr_inv_LCDMbg, np.array(GeffOverG_TraykovaL2L39_effect), '-', color=CB_color_cycle[2], label=r'$G_{\rm eff}/G_{\rm N}$')
axins = inset_axes(ax1, width="90%", height="90%", bbox_to_anchor=(.1, .175, .25, .66), bbox_transform=ax1.transAxes, loc='lower left')#zoomed_inset_axes(ax1, 2, loc=1) # zoom = 0.5
#ax1.add_patch(plt.Rectangle((.1, .175), .25, .66, ls="--", ec="c", fc="none", transform=ax1.transAxes))
axins.axhline(y=1., linestyle=':', color='black')
axins.semilogx(z_arr_inv_LCDMbg, E_TraykovaL2L39_modbg_effect, '-', color=CB_color_cycle[7])
axins.semilogx(z_arr_inv_LCDMbg, E_prime_E_TraykovaL2L39_modbg_effect, '-', color=CB_color_cycle[0])
axins.semilogx(z_arr_inv_LCDMbg, np.array(GeffOverG_TraykovaL2L39_effect), '-', color=CB_color_cycle[2])

# sub region of the original image
x1, x2, y1, y2 = 0., 10., 0.985, 1.005
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#axins.set_xticks("")
#axins.set_yticks("")

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax1, axins, loc1=3, loc2=4, fc="none", ec="0.5")

plt.draw()

ax2.axhline(y=1., linestyle=':', color='black')
#ax2.semilogx(z_arr_inv_LCDMbg, D_growth_arr_inv_TraykovaL2L39_Omm_effect, 'r-.', label=r'$D_E$')
#ax2.semilogx(z_arr_inv_LCDMbg, D_growth_arr_inv_TraykovaL2L39_EpE_effect, 'b-.', label=r'$D_{E^{\prime}/E}$')
ax2.semilogx(z_arr_inv_LCDMbg, D_growth_arr_inv_TraykovaL2L39_modbg_effect, '--', color=CB_color_cycle[5], label=r'$D_{\rm QCDM}$')#=D_E+D_{E^{\prime}/E}$')
ax2.semilogx(z_arr_inv_LCDMbg, D_growth_arr_inv_TraykovaL2L39_Geff_effect, '--', color=CB_color_cycle[2], label=r'$D_{G_{\rm eff}}$')
ax2.semilogx(z_arr_inv_LCDMbg, D_growth_arr_inv_TraykovaL2L39_full_effect, '-', color=CB_color_cycle[1], label=r'$D_{\rm ESS}=D_{\rm QCDM}+D_{G_{\rm eff}}$')
ax1.set_ylabel(r'$X^{\rm ESS}(z)/X^{\Lambda{\rm CDM}}(z)$')
ax2.set_ylabel(r'$D_1^{\rm ESS}(z)/D_1^{\Lambda{\rm CDM}}(z)$')
ax2.set_xlabel(r'$z$', labelpad=0.05)
ax1.legend(fontsize='small')
ax2.legend(fontsize='small')
ax1.set_xlim(0., 10.)
fig.set_size_inches(20, 10, forward=True)
plt.tight_layout(pad=0.1)
#plt.savefig('Plots/HiCOLA/Paper_plots/final_ESS_linear_growth.pdf', bbox_inches='tight')
#plt.show()


# GR ratio

fig = plt.figure()
plt.axhline(y=1., linestyle=':', color='black', label=r'$\Lambda$CDM')
plt.semilogx(Traykova_k_arr, FML_TraykovaL2_GR_ratio_L400_Np512_Nmesh1536_z0_P_arr, '-', color=CB_color_cycle[0], label='ESS-A')
plt.semilogx(Traykova_k_arr, FML_TraykovaL2L11_GR_ratio_L400_Np512_Nmesh1536_z0_P_arr, '-', color=CB_color_cycle[2], label='ESS-B')
plt.semilogx(Traykova_k_arr, FML_TraykovaL2L39_GR_ratio_L400_Np512_Nmesh1536_z0_P_arr, '-', color=CB_color_cycle[7], label='ESS-C')
plt.legend(fontsize='x-small')
plt.xlabel(r'$k\ [h/{\rm Mpc}]$', labelpad=0.05)
plt.ylabel(r'$P^{\rm ESS}(k, z=0)/P^{\Lambda{\rm CDM}}(k, z=0)$')
plt.xlim(0.02, 2.)
plt.ylim(0.99, 1.09)#1.1)
fig.set_size_inches(20, 10, forward=True)
plt.tight_layout(pad=0.1)
#plt.savefig('Plots/HiCOLA/Paper_plots/final_ESS_Pk_GR_ratio.pdf', bbox_inches='tight')
#plt.show()


# boost factor plot

fig, axarr = plt.subplots(nrows=2, ncols=2, sharey=False, sharex=True)
ax1 = axarr[0,0]
ax2 = axarr[0,1]
ax3 = axarr[1,0]
ax4 = axarr[1,1]

ax1.semilogx(FML_TraykovaL2_L400_Np512_Nmesh1536_z0_modbgeff_ratio_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_modbgeff_ratio_P_arr, '-', color=CB_color_cycle[0])
ax1.semilogx(FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_modbgeff_ratio_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_modbgeff_ratio_P_arr, '-', color=CB_color_cycle[2])
ax1.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_modbgeff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_modbgeff_ratio_P_arr, '-', color=CB_color_cycle[7])
ax1.set_ylabel(r'$\frac{P^{\rm QCDM}(k, z=0)}{P^{\Lambda{\rm CDM}}(k, z=0)}$', fontsize='x-large')
ax1.set_ylim(0.98, 1.001)

ax2.semilogx(FML_TraykovaL2_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_P_arr, '-', color=CB_color_cycle[0], label='ESS-A')
ax2.semilogx(FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_P_arr, '-', color=CB_color_cycle[2], label='ESS-B')
ax2.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_unscr5eff_ratio_P_arr, '-', color=CB_color_cycle[7], label='ESS-C')
ax2.set_ylabel(r'$\frac{P^{\rm lin\, ESS}(k, z=0)}{P^{\rm QCDM}(k, z=0)}$', fontsize='x-large')
ax2.set_ylim(0.99, 1.5)
ax2.legend(fontsize='small')

ax3.semilogx(FML_TraykovaL2_L400_Np512_Nmesh1536_z0_screeneff_ratio_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_screeneff_ratio_P_arr, '-', color=CB_color_cycle[0])
ax3.semilogx(FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_screeneff_ratio_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_screeneff_ratio_P_arr, '-', color=CB_color_cycle[2])
ax3.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_screeneff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_screeneff_ratio_P_arr, '-', color=CB_color_cycle[7])
ax3.set_ylabel(r'$\frac{P^{\rm ESS}(k, z=0)}{P^{\rm lin\, ESS}(k, z=0)}$', fontsize='x-large')
ax3.set_ylim(0.69, 1.01)

ax4.semilogx(FML_TraykovaL2_L400_Np512_Nmesh1536_z0_scr5eff_ratio_k_arr, FML_TraykovaL2_L400_Np512_Nmesh1536_z0_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[0])
ax4.semilogx(FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_scr5eff_ratio_k_arr, FML_TraykovaL2L11_L400_Np512_Nmesh1536_z0_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[2])
ax4.semilogx(FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_scr5eff_ratio_k_arr, FML_TraykovaL2L39_L400_Np512_Nmesh1536_z0_scr5eff_ratio_P_arr, '-', color=CB_color_cycle[7])
ax4.set_ylabel(r'$\frac{P^{\rm ESS}(k, z=0)}{P^{\rm QCDM}(k, z=0)}$', fontsize='x-large')
ax4.set_xlim(0.02, 2.)
ax4.set_ylim(0.999, 1.11)

ax3.set_xlabel(r'$k\ [h/{\rm Mpc}]$', labelpad=0.05)
ax4.set_xlabel(r'$k\ [h/{\rm Mpc}]$', labelpad=0.05)
fig.set_size_inches(20, 10, forward=True)
plt.tight_layout(pad=0.2)
#plt.savefig('Plots/HiCOLA/Paper_plots/final_ESS_Pk_effects_breakdown.pdf', bbox_inches='tight')
#plt.show()
