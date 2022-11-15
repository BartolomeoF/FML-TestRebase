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
import expression_builder as eb
import numerical_solver as ns
import sympy as sym
import sys
import itertools as it
import time
import os
from suppressor import *
import support as sp
from configobj import ConfigObj
from read_parameters import read_in_parameters
from argparse import ArgumentParser

to_exec = eb.declare_symbols()
exec(to_exec)
odeint_parameter_symbols = [E, phiprime, omegar, omegam]
 
parser = ArgumentParser(prog='Generate_Simulation_Input')
parser.add_argument('input_ini_filenames',nargs=2)

args = parser.parse_args()
print(args)
filenames = args.input_ini_filenames
Horndeski_path = filenames[0]
numerical_path = filenames[1]

model, K, G3, G4, parameters, mass_ratio_list, symbol_list, closure_declaration, cosmological_parameters, initial_conditions, sim_parameters, threshold_value, GR_flag = read_in_parameters(Horndeski_path, numerical_path)

[Omega_r0, Omega_m0, Omega_l0] = cosmological_parameters
[U0, phi_prime0] = initial_conditions
[Npoints, z_max, supp_flag] = sim_parameters


#---Create Horndeski functions---
###################COMMENT/UNCOMMENT
E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, fried_RHS_lambda, A_lambda, B2_lambda, \
    coupling_factor, alpha0_lambda, alpha1_lambda, alpha2_lambda, beta0_lambda, calB_lambda, calC_lambda = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list)
######################################


print('Horndeski functions -----------')
print(K)
print(type(K))
print(G3)
print(G4)
print('Horndeski parameters-----------')
print(parameters)
print('Symbol list-------')
print(symbol_list)
print(type(symbol_list[0]))
print('Lambdified functions-----------')
print(E_prime_E_lambda(1,1,1,1,1,1,1,1))

# #---Run solver and save output---COMMENT/UNCOMMENT
print(model+' model parameters, ' + str(symbol_list)+' = '+str(parameters))
print('Cosmological parameters are:')
print('Omega_m0 = '+str(Omega_m0))
print('Omega_r0 = '+str(Omega_r0))
a_arr,  UE_arr, UE_prime_UE_arr,UE_prime_arr2, phi_prime_arr, phi_primeprime_arr, Omega_r_arr, Omega_m_arr, Omega_DE_arr, Omega_l_arr, Omega_phi_arr, Omega_phi_diff_arr, Omega_r_prime_arr, Omega_m_prime_arr, Omega_l_prime_arr, A_arr, calB_arr, calC_arr, coupling_factor_arr, chioverdelta_arr = ns.run_solver_inv(Npoints, z_max, U0, phi_prime0, E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, A_lambda, fried_RHS_lambda, calB_lambda, calC_lambda, coupling_factor, closure_declaration, parameters, Omega_r0, Omega_m0, Omega_l0, symbol_list, odeint_parameter_symbols, supp_flag, threshold_value, GR_flag)
closure_arr = Omega_m_arr + Omega_r_arr + Omega_l_arr + Omega_phi_arr
Omega_DE_sum_arr = Omega_phi_arr + Omega_l_arr

print('Files for Hi-COLA numerical simulation being generated.')
###----Intermediate quantities-----
E_arr = np.array(UE_arr)/U0 #check whether COLA requires intermediates constructed with E rather than U!
E_prime_arr = np.array(UE_prime_arr2)/U0 #check whether COLA requires intermediates constructed with Eprime rather than Uprime!
##Note: E_prime_E is the same as U_prime_U, so that array does not need to be multiplied by anything.

directory = '../input/Horndeski'

filename_expansion = directory+'/%s_expansion.txt' %model
filename_force = directory+'/%s_force.txt' %model

print(filename_expansion)

sp.write_data_flex([a_arr,E_arr, UE_prime_UE_arr],filename_expansion)
sp.write_data_flex([a_arr,chioverdelta_arr,coupling_factor_arr],filename_force)
abs_directory = os.path.abspath(directory)
print('Files generated. Saved in "'+abs_directory+'".')


