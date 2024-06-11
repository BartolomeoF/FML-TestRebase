###################
# Loading modules #
###################

import numpy as np
import scipy.integrate as integrate
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from HiCOLA.Frontend.expression_builder import *
import sympy as sym
import sys
import itertools as it
import time
import os
from HiCOLA.Utilities.Other.suppressor import *


##########################
# Cosmological functions #
##########################

def comp_H_LCDM(z, Omega_r0, Omega_m0, H0):
    Omega_L0 = 1.-Omega_m0-Omega_r0
    H = H0*np.sqrt(Omega_m0*(1.+z)**3. + Omega_r0*(1.+z)**4. + Omega_L0)
    return H

def comp_E_LCDM(z, Omega_r0, Omega_m0):
    Omega_L0 = 1.-Omega_m0-Omega_r0
    E = np.sqrt(Omega_m0*(1.+z)**3. + Omega_r0*(1.+z)**4. + Omega_L0)
    return E

def comp_E_LCDM_DE(z, Omega_r0, Omega_m0):
    Omega_L0 = 1.-Omega_m0-Omega_r0
    E = np.sqrt(Omega_m0*(1.+z)**3. + Omega_r0*(1.+z)**4. + Omega_L0)
    return E

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

def comp_Omega_DE_LCDM(x, Omega_r0, Omega_m0):
    Omega_DE0 = 1. - Omega_m0 - Omega_r0
    term1 = Omega_r0*np.exp(-4.*x)+Omega_m0*np.exp(-3.*x)+Omega_DE0
    Omega_DE = Omega_DE0/term1
    return Omega_DE

def comp_E_prime_E_LCDM(x, Omega_r0, Omega_m0):
    Omega_DE0 = 1. - Omega_m0 - Omega_r0
    term1 = Omega_r0*np.exp(-4.*x)+Omega_m0*np.exp(-3.*x)+Omega_DE0
    term2 = 4.*Omega_r0*np.exp(-4.*x)+3.*Omega_m0*np.exp(-3.*x)
    E_prime_E = -0.5*term2/term1
    return E_prime_E

def comp_Omega_DE_prime_LCDM(E_prime_E, Omega_DE):
    Omega_DE_prime = -2.*E_prime_E*Omega_DE
    return Omega_DE_prime

def comp_alpha_M_propto_Omega_DE_LCDM(c_M, Omega_DE):
    alpha_M_DE = c_M*Omega_DE
    return alpha_M_DE

def comp_alpha_M_prime_propto_Omega_DE_LCDM(c_M, Omega_DE_prime):
    alpha_M_prime_DE = c_M*Omega_DE_prime
    return alpha_M_prime_DE

def alpha_M_int_propto_Omega_DE_LCDM(x, Omega_r0, Omega_m0, c_M):
    Omega_DE = comp_Omega_DE_LCDM(x, Omega_r0, Omega_m0)
    alpha_M_int = c_M * Omega_DE
    return alpha_M_int

def comp_Meffsq_x2_x1_propto_Omega_DE_LCDM(x1, x2, Omega_r0, Omega_m0, c_M):
    Meffsq_x2_x1 = np.exp(integrate.quad(alpha_M_int_propto_Omega_DE_LCDM, x1, x2, args=(Omega_r0, Omega_m0, c_M))[0])
    return Meffsq_x2_x1

def comp_Omega_r_prime(Omega_r, E, E_prime):
    E_prime_E = E_prime/E
    Omega_r_prime = -Omega_r*(4.+2.*E_prime_E)
    return Omega_r_prime

def comp_Omega_l_prime(Omega_l0, E, Eprime):
    return -2.*Omega_l0*Eprime/E/E/E

def comp_Omega_m_prime(Omega_m, E, E_prime):
    E_prime_E = E_prime/E
    Omega_m_prime = -Omega_m*(3.+2.*E_prime_E)
    return Omega_m_prime

#fried1 is ultimately the closure equation for the density parameters
def fried1(phi_prime, k1, g1, Omega_r, Omega_m, E, alpha_M, Ms_Mp, Meffsq_Mpsq):
    zer0 = (Omega_r+Omega_m)/Meffsq_Mpsq
    zer1a = 3.*Meffsq_Mpsq/Ms_Mp/Ms_Mp
    zer1b = 0.5*k1*phi_prime**2.
    zer1c = 3.*g1*Ms_Mp*E**2.*phi_prime**3.
    zer1 = (zer1b+zer1c)/zer1a
    zer2 = -alpha_M
    zer = zer0 + zer1 + zer2 - 1.
    return zer


def fried_RHS_wrapper(cl_variable, cl_declaration, fried_RHS_lambda, E, phi_prime, Omega_r, Omega_m, Omega_l, parameters):
    argument_no = 5 + len(parameters)
    if cl_declaration[1] > argument_no -1:
        raise Exception('Invalid declaration - there is no valid argument index for the declaration')
    if cl_declaration[0] == 'odeint_parameters':
        if cl_declaration[1] == 0:
            return fried_RHS_lambda(cl_variable,phi_prime,Omega_r,Omega_m,Omega_l, f_phi, *parameters) #Closure used to compute E0
        if cl_declaration[1] == 1:
            return fried_RHS_lambda(E,cl_variable,Omega_r,Omega_m,Omega_l,*parameters) #Closure used to compute phi0
        if cl_declaration[1] == 2:
            return fried_RHS_lambda(E,phi_prime,cl_variable,Omega_m,Omega_l,*parameters) #Closure used to compute Omega_r0
        if cl_declaration[1] == 3:
            return fried_RHS_lambda(E,phi_prime,Omega_r,cl_variable,Omega_l,*parameters) #Closure used to compute Omega_m0
        if cl_declaration[1] == 4:
            return fried_RHS_lambda(E,phi_prime,Omega_r,Omega_m,cl_variable,*parameters) #Closure used to compute Omega_l0
    if cl_declaration[0] == 'parameters':
        parameters[cl_declaration[1]] = cl_variable
        return fried_RHS_lambda(E,phi_prime,Omega_r,Omega_m,Omega_l,*parameters)
    else:
        raise Exception('Invalid string in declaration list. Must be either \'odeint_parameters\' or \'parameters\'')



def comp_param_close(fried_closure_lambda, cl_declaration, E0, phi_prime0, Omega_r0, Omega_m0, Omega_l0,parameters): 
    cl_guess = 1.0 #this may need to be changed depending on what is being solved for through closure, if fsolve has trouble
    if cl_declaration[0] == 'odeint_parameters':
        if cl_declaration[1] == 0:
            cl_guess = E0
        if cl_declaration[1] == 1:
            cl_guess = phi_prime0
        if cl_declaration[1] == 2:
            cl_guess = Omega_r0
        if cl_declaration[1] == 3:
            cl_guess = Omega_m0
        if cl_declaration[1] ==4:
            cl_guess = Omega_l0
    if cl_declaration[0] == 'parameters':
        cl_guess = parameters[cl_declaration[1]]
    cl_variable,fsolvedict,fsolveier,fsolvemsg = fsolve(fried_RHS_wrapper, cl_guess, args=(cl_declaration, fried_closure_lambda, E0, phi_prime0, Omega_r0,Omega_m0, Omega_l0,parameters), xtol=1e-6,full_output=True) 
    cl_variable = cl_variable[0]
    return cl_variable

def comp_primes(x, Y, E0, Omega_r0, Omega_m0, Omega_l0, E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, A_lambda, cl_declaration, parameters,threshold=1e-3,GR_flag=False): #x, Y swapped for solve_ivp ###ADD LAMBDA FUNCTION AS ARGUMENT###


    phi_primeY, EUY, Omega_rY, Omega_mY, Omega_lY = Y
    A_value = A_lambda(EUY,phi_primeY,*parameters)
    if A_value - abs(A_value) == 0:
        A_sign = 1.
    elif A_value - abs(A_value) != 0:
        A_sign = -1.


    if (abs(A_value) >= threshold and GR_flag==False) or (threshold==0. and GR_flag==False):
        E_prime_E_evaluated = E_prime_E_lambda(EUY,phi_primeY,Omega_rY,Omega_lY,*parameters)
        E_prime_evaluated = E_prime_E_evaluated*EUY
        phi_primeprime_evaluated = phi_primeprime_lambda(EUY,E_prime_evaluated,phi_primeY,*parameters)
    if (abs(A_value) < threshold and GR_flag==False):
        E_prime_E_evaluated = E_prime_E_safelambda(EUY,phi_primeY,Omega_rY,Omega_lY, threshold,A_sign,*parameters)
        E_prime_evaluated = E_prime_E_evaluated*EUY
        phi_primeprime_evaluated = phi_primeprime_safelambda(EUY,E_prime_evaluated,phi_primeY,threshold,A_sign,*parameters)
    if GR_flag==True:
        E_prime_E_evaluated = comp_E_prime_E_LCDM(x,Omega_r0,Omega_m0)
        E_prime_evaluated = E_prime_E_evaluated*EUY
        phi_primeprime_evaluated = 0.
    if cl_declaration[0] == 'odeint_parameters': #usually indicates dS approach, so we must convert U back to E, since this is what the Omega_prime functions use
        EY = EUY/E0
        EYprime = E_prime_evaluated/E0
    if cl_declaration[0] == 'parameters': #usually indicates 'today' approach, no need to change the Hubble variable, it is already E
        EY = EUY
        EYprime = E_prime_evaluated
    Omega_r_prime = comp_Omega_r_prime(Omega_rY, EY, EYprime)
    Omega_m_prime = comp_Omega_m_prime(Omega_mY, EY, EYprime)
    Omega_l_prime = comp_Omega_l_prime(Omega_l0,EY, EYprime)
    return [phi_primeprime_evaluated, E_prime_evaluated, Omega_r_prime, Omega_m_prime, Omega_l_prime]

def chi_over_delta(a_arr, E_arr, calB_arr, calC_arr, Omega_m0): #the E_arr is actual E, not U! Convert U_arr to E_arr!
    chioverdelta = np.array(calB_arr)*np.array(calC_arr)*Omega_m0/np.array(E_arr)/np.array(E_arr)/np.array(a_arr)/np.array(a_arr)/np.array(a_arr)
    return chioverdelta


def run_solver(read_out_dict):


    [Omega_r0, Omega_m0, Omega_l0] = read_out_dict['cosmological_parameters']
    [Hubble0, phi_prime0] = read_out_dict['initial_conditions']
    [Npoints, z_max, suppression_flag, threshold, GR_flag] = read_out_dict['simulation_parameters']
    parameters = read_out_dict['Horndeski_parameters']

    E_prime_E_lambda = read_out_dict['E_prime_E_lambda']
    E_prime_E_safelambda = read_out_dict['E_prime_E_safelambda']
    phi_primeprime_lambda = read_out_dict['phi_primeprime_lambda']
    phi_primeprime_safelambda = read_out_dict['phi_primeprime_safelambda']
    omega_phi_lambda = read_out_dict['omega_phi_lambda']
    A_lambda = read_out_dict['A_lambda']
    fried_RHS_lambda = read_out_dict['fried_RHS_lambda']
    calB_lambda = read_out_dict['calB_lambda']
    calC_lambda = read_out_dict['calC_lambda']
    coupling_factor = read_out_dict['coupling_factor']

    parameter_symbols = read_out_dict['symbol_list']
    odeint_parameter_symbols = read_out_dict['odeint_parameter_symbols']
    cl_declaration = read_out_dict['closure_declaration']

    z_final = 0.
    x_ini = np.log(1./(1.+z_max))
    x_final = np.log(1./(1.+z_final))
    x_arr = np.linspace(x_ini, x_final, Npoints)
    a_arr = [np.exp(x) for x in x_arr]
    x_arr_inv = x_arr[::-1]
    a_arr_inv = a_arr[::-1]



    if GR_flag is True:
        phi_prime0 = 0.

    cl_var = comp_param_close(fried_RHS_lambda, cl_declaration, Hubble0, phi_prime0, Omega_r0, Omega_m0, Omega_l0, parameters)

    if cl_declaration[0] == 'odeint_parameters':
        if cl_declaration[1] == 0:
            Hubble0_closed = cl_var
            Y0 = [phi_prime0,Hubble0_closed,Omega_r0,Omega_m0, Omega_l0]
        if cl_declaration[1] == 1:
            phi_prime0_closed = cl_var
            if 1.-Omega_r0 - Omega_m0 == Omega_l0:
                    phi_prime0_closed = 0.
            Y0 = [phi_prime0_closed,Hubble0,Omega_r0,Omega_m0, Omega_l0]
        if cl_declaration[1] == 2:
            Omega_r0_closed = cl_var
            Y0 =[phi_prime0,Hubble0,Omega_r0_closed,Omega_m0, Omega_l0]
        if cl_declaration[1] == 3:
            Omega_m0_closed = cl_var
            Y0 = [phi_prime0,Hubble0,Omega_r0,Omega_m0_closed, Omega_l0]

    if cl_declaration[0] == 'parameters':
        parameters[cl_declaration[1]] = cl_var
        Y0 = [phi_prime0,Hubble0,Omega_r0,Omega_m0,Omega_l0]


    #if suppression_flag is True:
    #    with stdout_redirected():
    #        ans = odeint(comp_primes, Y0, x_arr_inv, args=(Hubble0, Omega_r0, Omega_m0, Omega_l0, E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, A_lambda, cl_declaration, parameters,threshold,GR_flag), tfirst=True)#, hmax=hmaxv) #k1=-6, g1 = 2
    #else:
    #    ans = odeint(comp_primes, Y0, x_arr_inv, args=(Hubble0, Omega_r0, Omega_m0, Omega_l0, E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, A_lambda, cl_declaration, parameters,threshold,GR_flag), tfirst=True)#, hmax=hmaxv)

    ans = solve_ivp(comp_primes, t_eval=x_arr_inv, t_span=(x_arr_inv[0],x_arr_inv[-1]), y0=Y0, args=(Hubble0, Omega_r0, Omega_m0, Omega_l0, E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, A_lambda, cl_declaration, parameters,threshold,GR_flag), method='DOP853') #.y added to ans when variable is called if found using solve_ivp instead of odeint

    #slice indices swapped if using solve_ivp instead of odeint
    phi_prime_arr = ans.y[0,:] 
    Hubble_arr = ans.y[1,:]
    Omega_r_arr = ans.y[2,:]
    Omega_m_arr = ans.y[3,:]
    Omega_l_arr = ans.y[4,:]

    E_prime_E_LCDM_arr = [comp_E_prime_E_LCDM(xv, Omega_r0, Omega_m0) for xv in x_arr_inv]
    Omega_DE_LCDM_arr = [comp_Omega_DE_LCDM(xv, Omega_r0, Omega_m0) for xv in x_arr_inv]
    Omega_DE_prime_LCDM_arr = [comp_Omega_DE_prime_LCDM(E_prime_E_LCDMv, Omega_DE_LCDMv) for E_prime_E_LCDMv, Omega_DE_LCDMv in zip(E_prime_E_LCDM_arr, Omega_DE_LCDM_arr)]

    E_prime_E_arr = []
    phi_primeprime_arr = []
    for Omega_rv, Omega_lv, Ev, phi_primev in zip(Omega_r_arr, Omega_l_arr, Hubble_arr, phi_prime_arr):
        E_prime_E_arr.append(E_prime_E_lambda(Ev,phi_primev,Omega_rv, Omega_lv, *parameters))
    Hubble_prime_arr = [E_prime_Ev*Ev for E_prime_Ev, Ev in zip(E_prime_E_arr, Hubble_arr)]
    for Omega_rv, Ev, E_primev, phi_primev in zip(Omega_r_arr, Hubble_arr, Hubble_prime_arr, phi_prime_arr):
        phi_primeprime_arr.append(phi_primeprime_lambda(Ev,E_primev,phi_primev,*parameters))

    A_arr = []
    for Ev, phi_primev,  in zip(Hubble_arr,phi_prime_arr):
        A_arr.append(A_lambda(Ev, phi_primev, *parameters))

    Omega_phi_arr = []
    for Ev, phiprimev in zip(Hubble_arr,phi_prime_arr):
        Omega_phi_arr.append(omega_phi_lambda(Ev,phiprimev,*parameters))


    Omega_DE_arr = []
    Omega_phi_diff_arr = []
    Omega_r_prime_arr = []
    Omega_m_prime_arr= []
    Omega_l_prime_arr = []
    for Ev, E_primev, Omega_rv, Omega_mv, Omega_lv in zip(Hubble_arr, Hubble_prime_arr, Omega_r_arr,Omega_m_arr, Omega_l_arr):
        Omega_DE_arr.append(1. - Omega_rv - Omega_mv)
        Omega_phi_diff_arr.append(1. - Omega_rv - Omega_mv - Omega_lv)
        Omega_r_prime_arr.append(comp_Omega_r_prime(Omega_rv, Ev, E_primev))
        Omega_m_prime_arr.append(comp_Omega_m_prime(Omega_mv, Ev, E_primev))
        Omega_l_prime_arr.append(comp_Omega_l_prime(Omega_l0,Ev, E_primev))


    array_output = []
    for i in [a_arr_inv, Hubble_arr, E_prime_E_arr, Hubble_prime_arr, phi_prime_arr,  phi_primeprime_arr, Omega_r_arr, Omega_m_arr, Omega_DE_arr, Omega_l_arr, Omega_phi_arr, Omega_phi_diff_arr, Omega_r_prime_arr, Omega_m_prime_arr, Omega_l_prime_arr, A_arr]:
        i = np.array(i)
        array_output.append(i)
    [a_arr_inv, Hubble_arr, E_prime_E_arr, Hubble_prime_arr, phi_prime_arr,  phi_primeprime_arr, Omega_r_arr, Omega_m_arr, Omega_DE_arr, Omega_l_arr, Omega_phi_arr, Omega_phi_diff_arr, Omega_r_prime_arr, Omega_m_prime_arr, Omega_l_prime_arr, A_arr] = array_output

    calB_arr = []
    calC_arr = []
    coupling_factor_arr = []
    for UEv, UEprimev, phiprimev, phiprimeprimev in zip(Hubble_arr, Hubble_prime_arr, phi_prime_arr, phi_primeprime_arr):
        calB_arr.append(calB_lambda(UEv,UEprimev,phiprimev,phiprimeprimev, *parameters))
        calC_arr.append(calC_lambda(UEv,UEprimev,phiprimev,phiprimeprimev, *parameters))
        coupling_factor_arr.append(coupling_factor(UEv,UEprimev,phiprimev,phiprimeprimev,*parameters))

    E_arr = Hubble_arr/Hubble0
    chioverdelta_arr = chi_over_delta(a_arr_inv, E_arr, calB_arr, calC_arr, Omega_m0)

    solution_arrays = {'a':a_arr_inv, 'Hubble':Hubble_arr, 'Hubble_prime':Hubble_prime_arr,'E_prime_E':E_prime_E_arr,'scalar_prime':phi_prime_arr,'scalar_primeprime':phi_primeprime_arr}
    cosmological_density_arrays = {'omega_m':Omega_m_arr,'omega_r':Omega_r_arr,'omega_l':Omega_l_arr,'omega_phi':Omega_phi_arr, 'omega_DE':Omega_DE_arr}
    cosmo_density_prime_arrays = {'omega_m_prime':Omega_m_prime_arr,'omega_r_prime':Omega_r_prime_arr,'omega_l_prime':Omega_l_prime_arr}
    force_quantities = {'A':A_arr, 'calB':calB_arr, 'calC':calC_arr, 'coupling_factor':coupling_factor_arr, 'chi_over_delta':chioverdelta_arr}
    result = {}

    for i in [solution_arrays, cosmological_density_arrays, cosmo_density_prime_arrays,force_quantities]:
        result.update(i)
    result.update({'closure_value':cl_var})

    return result