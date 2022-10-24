#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 01:55:07 2021

@author: ashimsg
"""

###################
# Loading modules #
###################

import numpy as np
import scipy.integrate as integrate
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from expression_builder import *
import numerical_solver as nta
import sympy as sym
import sys
import itertools as it
import time
import os
from suppressor import *
import support as sp

K,G3,G4 =None, None, None
dashed_line = '-'

###########################
# Cosmological parameters #
###########################

### values based on fit to WMAP in table 1, 1306.3219
Omega_r0h2 = 4.28e-5
Omega_b0h2 = 0.02196
Omega_c0h2 = 0.1274
h_test = 0.7307

Omega_r0 = Omega_r0h2/h_test/h_test 
Omega_m0 = (Omega_b0h2 + Omega_c0h2)/h_test/h_test
Omega_DE0 = 1. - Omega_r0 - Omega_m0



################
# Running Code 
################

supp_flag = False
Npoints = 1000
z_ini_inv =  1000.

M_pG4_test, M_KG4_test, M_G3s_test, M_sG4_test, M_G3G4_test, M_Ks_test, M_gp_test = 1., 1., 1., 1., 1., 1., 1. #maybe scale everything to M_p instead   
mass_ratio_list = [M_pG4_test, M_KG4_test, M_G3s_test, M_sG4_test, M_G3G4_test, M_Ks_test, M_gp_test]


#---Do not change, this remains the same for any model----
a, E, Eprime, phi, phiprime, phiprimeprime, X,  M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, M_gp, omegar, omegam, omegal, f, Theta = sym.symbols("a E Eprime phi phiprime phiprimeprime X M_{pG4} M_{KG4} M_{G3s} M_{sG4} M_{G3G4} M_{Ks} M_{gp} Omega_r Omega_m Omega_l f Theta")
odeint_parameter_symbols = [E, phiprime, omegar, omegam]
#------------------------

#---Add/remove according to model being used----
k1, g31, g32, k2, g4 = sym.symbols("k_1 g_{31} g_{32} k_2 g_4")

symbol_list = [k1,k2,g31,g32]
#---Numerical constants in model---

Lambda_2 = 1.
Lambda_3 = 1.

g4 = 0.

directory = '../input/Horndeski/'
if not os.path.exists(directory):
    os.makedirs(directory)




##### SELECT MODEL #################
model='ESS'
case='A'

###OR WRITE YOUR OWN K, G3, G4 BELOW

# K  = X 
# G3 = X**0.5
# G4 = 0.5
# EdS = 1.0
# parameters=[1,1,1,1]
# Omega_l0=1
###################################

if (model=='cubic_Galileon' or model=='cubic' or model=='Cubic Galileon' or model=='cG'):
    K = k1*X
    G3 = g31*M_sG4_test*X/M_pG4_test
    G4 = 0.5 +g4*M_gp_test*M_sG4_test*phi/M_pG4_test
elif (model=='ESS'):
    K = k1*X + k2*X**2./(Lambda_2**4.)
    G3 = (g31*X + g32*X**2./(Lambda_2**4.))/(Lambda_3**3.)
    G4 = 0.5 +g4*M_gp_test*M_sG4_test*phi/M_pG4_test
##############################################################################

closure_declaration = ['odeint_parameters',1] #decide which parameter should be fixed via closure



#---Create Horndeski functions---
###################COMMENT/UNCOMMENT
E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, fried_RHS_lambda, A_lambda, B2_lambda, \
    coupling_factor, alpha0_lambda, alpha1_lambda, alpha2_lambda, beta0_lambda, calB_lambda, calC_lambda = create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list)
######################################





 
###################################
'''
The following models and cases reproduce the data required to generate the results of the first Hi-COLA paper.
'''

if  model=='ESS':
    symbol_list = [k1,k2,g31,g32]
    E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, fried_RHS_lambda, A_lambda, B2_lambda, \
    coupling_factor, alpha0_lambda, alpha1_lambda, alpha2_lambda, beta0_lambda, calB_lambda, calC_lambda = create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list)
    if case=='A':
        filename_chicouple = directory+'ESS-A_force.txt'
        filename_expansion = directory+'ESS-A_expansion.txt'
        [EdS, f_value, k1_dSseed, g31_dSseed] = [0.8116666666666665, 0.125, -2.6315789473684177, -44.73684210526316]
    elif case=='B':
        filename_chicouple = directory+'ESS-B_force.txt'
        filename_expansion = directory+'ESS-B_expansion.txt'
        [EdS, f_value, k1_dSseed, g31_dSseed] = [0.7505263157894736, 0.3684210526315789, -4.526315789473684, -42.10526315789474]
    elif case=='C':
        filename_chicouple = directory+'ESS-C_force.txt'
        filename_expansion = directory+'ESS-C_expansion.txt'
        [EdS, f_value, k1_dSseed, g31_dSseed] = [0.7126315789473684, 0.42105263157894735, -1.9210526315789473, -46.31578947368421]
    else:
        raise Exception("ESS case not recognised, please either use case strings A, B or C or define a new case.")
    Omega_l0 = (1.-f_value)*Omega_DE0
    alpha_param = 1. - Omega_l0/EdS/EdS
    k1_dS, g31_dS = k1_dSseed*alpha_param, g31_dSseed*alpha_param
    k2_dS = -2.*k1_dS - 12.*(1-Omega_l0/EdS/EdS)
    g32_dS = 0.5*(1. - (Omega_l0/EdS/EdS + k1_dS/6. + k2_dS/4. + g31_dS))
    parameters = [k1_dS, k2_dS, g31_dS, g32_dS]
elif (model=='cG' or model=='cubic Galileon' or model=='Cubic Galileon'):
    symbol_list = [k1,g31]
    E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, fried_RHS_lambda, A_lambda, B2_lambda, \
    coupling_factor, alpha0_lambda, alpha1_lambda, alpha2_lambda, beta0_lambda, calB_lambda, calC_lambda = create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list)
    if (case<0.) or (case >1.):
        raise Exception("case, which in cG means f_phi, must be between 0 and 1 (inclusive).")
    filename_chicouple = directory+'cG-f'+str(case)+'_force.txt'
    filename_expansion = directory+'cG-f'+str(case)+'_expansion.txt'
    k1_dSseed = -6.
    g31_dSseed = 2.
    f_value = case
    E_dS_max_guess = 0.95
    almost = 1e-6
    EdS = nta.comp_E_dS_max(E_dS_max_guess, Omega_r0, Omega_m0, f_value, almost, fried_RHS_lambda)
    print("EdS is "+(str(EdS))) 
    Omega_l0 = (1.-f_value)*Omega_DE0
    alpha_param = 1. - Omega_l0/EdS/EdS
    k1_dS, g31_dS = k1_dSseed*alpha_param, g31_dSseed*alpha_param
    parameters = [k1_dS, g31_dS]    
elif model == 'user':
    print("Horndeki functions K, G3, G4 specified by user.")
    filename_chicouple = directory+'Hi-COLA_force.txt'
    filename_expansion = directory+'Hi-COLA_expansion.txt'
else:
    raise Exception("Model not recognised, please use ESS or Cubic Galileon (cG), or define your own.")
if len(symbol_list) != len(parameters):
    raise Exception('The number of parameter symbols should match the number of numerical model parameter values.')

####################################################################     



   

U0 = 1./EdS
phi_prime0 = 0.9 ##depending on closure choice, this might be guess for fsolve
threshold_value = 0.


# #------------------------------------
GR_flag = False
# param_check_arr = np.array(parameters)
# check_counter = 0
# for i in param_check_arr:
#     if i == 0.:
#         check_counter +=1
# if check_counter ==len(parameters):
#     GR_flag = True
# elif check_counter != len(parameters):
#     GR_flag = False
# print('GR flag is '+str(GR_flag))
# #--------------------------------

# #---Run solver and save output---COMMENT/UNCOMMENT
print(model+' model parameters, ' + str(symbol_list)+' = '+str(parameters))
print('Cosmological parameters are:')
print('Omega_m0 = '+str(Omega_m0))
print('Omega_r0 = '+str(Omega_r0))
a_arr,  UE_arr, UE_prime_UE_arr,UE_prime_arr2, phi_prime_arr, phi_primeprime_arr, Omega_r_arr, Omega_m_arr, Omega_DE_arr, Omega_l_arr, Omega_phi_arr, Omega_phi_diff_arr, Omega_r_prime_arr, Omega_m_prime_arr, Omega_l_prime_arr, A_arr, calB_arr, calC_arr, coupling_factor_arr, chioverdelta_arr = nta.run_solver_inv(Npoints, z_ini_inv, U0, phi_prime0, E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, A_lambda, fried_RHS_lambda, calB_lambda, calC_lambda, coupling_factor, closure_declaration, parameters, Omega_r0, Omega_m0, Omega_l0, symbol_list, odeint_parameter_symbols, supp_flag, threshold_value, GR_flag)
closure_arr = Omega_m_arr + Omega_r_arr + Omega_l_arr + Omega_phi_arr
Omega_DE_sum_arr = Omega_phi_arr + Omega_l_arr

print('Files for Hi-COLA numerical simulation being generated.')
###----Intermediate quantities-----
E_arr = np.array(UE_arr)*EdS #check whether COLA requires intermediates constructed with E rather than U!
E_prime_arr = np.array(UE_prime_arr2)*EdS #check whether COLA requires intermediates constructed with Eprime rather than Uprime!
##Note: E_prime_E is the same as U_prime_U, so that array does not need to be multiplied by anything.








sp.write_data_flex([a_arr,E_arr, UE_prime_UE_arr],filename_expansion)
sp.write_data_flex([a_arr,chioverdelta_arr,coupling_factor_arr],filename_chicouple)
abs_directory = os.path.abspath(directory)
print('Files generated. Saved in "'+abs_directory+'".')


