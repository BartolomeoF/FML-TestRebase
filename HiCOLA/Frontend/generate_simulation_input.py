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
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from HiCOLA.Frontend import expression_builder as eb
import HiCOLA.Frontend.numerical_solver as ns
import sympy as sym
import sys
import itertools as it
import time
import os
from HiCOLA.Utilities.Other import support as sp
from configobj import ConfigObj
from HiCOLA.Frontend.read_parameters import read_in_parameters
from argparse import ArgumentParser

to_exec = eb.declare_symbols()
exec(to_exec)

parser = ArgumentParser(prog='Generate_Simulation_Input')
parser.add_argument('input_ini_filenames',nargs=2)

args = parser.parse_args()
print(args)
filenames = args.input_ini_filenames
Horndeski_path = filenames[0]
numerical_path = filenames[1]

read_out_dict = read_in_parameters(Horndeski_path, numerical_path)
odeint_parameter_symbols = [E, phi, phiprime, omegar, omegam]
read_out_dict.update({'odeint_parameter_symbols':odeint_parameter_symbols})

model = read_out_dict['model_name']
K = read_out_dict['K']
G3 = read_out_dict['G3']
G4 = read_out_dict['G4']
cosmology_name = read_out_dict['cosmo_name']
[Omega_r0, Omega_m0, Omega_l0] = read_out_dict['cosmological_parameters']
[U0, phi0, phi_prime0] = read_out_dict['initial_conditions']
[Npoints, z_max, supp_flag, threshold_value, GR_flag] = read_out_dict['simulation_parameters']
parameters = read_out_dict['Horndeski_parameters']
mass_ratio_list = read_out_dict['mass_ratio_list']
symbol_list = read_out_dict['symbol_list']
closure_declaration = read_out_dict['closure_declaration']

closure_dictionary = {'odeint_parameters':odeint_parameter_symbols, 'parameters':symbol_list}

lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list)
read_out_dict.update(lambdified_functions)


print('Horndeski functions --------------')
print(f'K = {K}')
print(f'G_3 = {G3}')
print(f'G_4 = {G4}')
print('Horndeski parameters--------------')

# #---Run solver and save output---COMMENT/UNCOMMENT
print(model+' model parameters, ' + str(symbol_list)+' = '+str(parameters))
print('Cosmological parameters-----------')
print('Omega_m0 = '+str(Omega_m0))
print('Omega_r0 = '+str(Omega_r0))
print(f'Omega_lambda0 = {Omega_l0}')
print('Initial conditions----------------')
print(f'Hubble0 = {U0}')
print(f'scalar0 = {phi0}')
print(f'scalar_prime0 = {phi_prime0}')

background_quantities = ns.run_solver(read_out_dict)
a_arr = background_quantities['a']
UE_arr = background_quantities['Hubble']
UE_prime_arr = background_quantities['Hubble_prime']
UE_prime_UE_arr = background_quantities['E_prime_E']
coupling_factor_arr = background_quantities['coupling_factor']
chioverdelta_arr = background_quantities['chi_over_delta']
Omega_m_arr = background_quantities['omega_m']
Omega_r_arr = background_quantities['omega_r']
Omega_lambda_arr = background_quantities['omega_l']
Omega_phi_arr = background_quantities['omega_phi']
phi_arr = background_quantities['scalar']
phi_prime_arr = background_quantities['scalar_prime']
phi_primeprime_arr = background_quantities['scalar_primeprime']

closure_variable = str(closure_dictionary[closure_declaration[0]][closure_declaration[1]])
closure_value = str(background_quantities['closure_value'])
print(f'Closure parameter is {closure_variable} = {closure_value}' )
print('(note: therefore one of the initial conditions or Horndeski model parameters printed above was the guess value)')

#----Compute alphas----
alphas_arr = ns.comp_alphas(read_out_dict, background_quantities)
M_star_sqrd_arr = alphas_arr[0]
alpha_M_arr = alphas_arr[1]
alpha_B_arr = alphas_arr[2]
alpha_K_arr = alphas_arr[3]

#----Compute Dark Energy EoS----
DE_arr = ns.comp_w_DE(background_quantities)#to change compute method remove 2 and read_out_dict
w_DE_arr = DE_arr[0]
P_DE_arr = DE_arr[1]
rho_DE_arr = DE_arr[2]

#----Checking stability conditions----
Q_S_arr, c_s_sq_arr = ns.comp_stability(read_out_dict, background_quantities)

#----Alpha parameterisation----

#first parameterisation
alpha_M01 = 0.
alpha_B01 = 0.288
alpha_K01 = 0.864
print('First parameterisation------------')
print('alpha_X = alpha_X0*(1 - Omega_m - Omega_r)/(1 - Omega_m0 - Omega_r0)')
print("alpha_M0, alpha_B0, alpha_K0 = {}, {}, {}".format(alpha_M01, alpha_B01, alpha_K01))
alpha_X01 = np.array([[alpha_M01],[alpha_B01],[alpha_K01]])
alphas_param1_arr = ns.alpha_X1(alpha_X01, read_out_dict, background_quantities)
alpha_M_param1_arr = alphas_param1_arr[0]
alpha_B_param1_arr = alphas_param1_arr[1]
alpha_K_param1_arr = alphas_param1_arr[2]

#second parameterisation
alpha_X = np.concatenate((alpha_M_arr, alpha_B_arr, alpha_K_arr))
(alpha_M02, alpha_B02, alpha_K02, q), junk = curve_fit(ns.alpha_X2, a_arr, alpha_X, bounds=([-1,-1,-1,2],[1,1,1,6]))
print('Second parameterisation-----------')
print('alpha_X = alpha_X0*(1 + z)^(-q)')
print("alpha_M0, alpha_B0, alpha_K0 = {}, {}, {}".format(alpha_M02, alpha_B02, alpha_K02))
print("q = ", q)
alpha_M_param2_arr = ns.alpha_M3(a_arr, alpha_M02, q)
alpha_B_param2_arr = ns.alpha_B3(a_arr, alpha_B02, q)
alpha_K_param2_arr = ns.alpha_K3(a_arr, alpha_K02, q)

#third parameterisation
(alpha_M03, q_M), junk = curve_fit(ns.alpha_M3, a_arr, alpha_M_arr, bounds=([-1,2],[1,6]))
(alpha_B03, q_B), junk = curve_fit(ns.alpha_B3, a_arr, alpha_B_arr, bounds=([-1,2],[1,6]))
(alpha_K03, q_K), junk = curve_fit(ns.alpha_K3, a_arr, alpha_K_arr, bounds=([-1,2],[1,6]))
print('Third parameterisation------------')
print('alpha_X = alpha_X0*(1 + z)^(-q_X)')
print("alpha_M0, alpha_B0, alpha_K0 = {}, {}, {}".format(alpha_M03, alpha_B03, alpha_K03))
print("q_M, q_B, q_K = {}, {}, {}".format(q_M, q_B, q_K))
alpha_M_param3_arr = ns.alpha_M3(a_arr, alpha_M03, q_M)
alpha_B_param3_arr = ns.alpha_B3(a_arr, alpha_B03, q_B)
alpha_K_param3_arr = ns.alpha_K3(a_arr, alpha_K03, q_K)

print('Files for Hi-COLA numerical simulation being generated.')
###----Intermediate quantities-----
##Note: U = E/E_dS
## U0 = 1/E_dS
E_arr = np.array(UE_arr) #check whether you require intermediates constructed with E rather than U!
E_prime_arr = np.array(UE_prime_arr)#check whether backend requires intermediates constructed with Eprime rather than Uprime!
##Note: E_prime_E is the same as U_prime_U, so that array does not need to be multiplied by anything.

directory = read_out_dict['output_directory']

if not os.path.exists(directory):
    os.makedirs(directory)

filename_expansion = directory+f'/{model}_{cosmology_name}_expansion.txt'
filename_force = directory+f'/{model}_{cosmology_name}_force.txt'
filename_stability = directory+f'/{model}_{cosmology_name}_stability.txt'
filename_properties = directory+f'/{model}_{cosmology_name}_properties.txt'

abs_directory = os.path.abspath(directory)
loop_counter = 0
while ( os.path.exists(filename_expansion) or os.path.exists(filename_force) or os.path.exists(filename_properties) ) and loop_counter < 100:
    loop_counter += 1
    filename_expansion = sp.renamer(filename_expansion)
    filename_force = sp.renamer(filename_force)
    filename_properties = sp.renamer(filename_properties)
if loop_counter >= 100:
    raise Exception("Counter for file renaming loop excessively high, consider changing expansion, force and properties output file names.")
if loop_counter != 0:
    print(f"Warning: expansion, force or properties file with same name found in \"{abs_directory}\", new filenames are \n expansion: {filename_expansion} \n force:{filename_force} \n properties:{filename_properties}")

loop_counter = 0
while os.path.exists(filename_stability) and loop_counter < 100:
    loop_counter += 1
    filename_stability = sp.renamer(filename_stability)
if loop_counter >= 100:
    raise Exception("Counter for file renaming loop excessively high, consider changing stability output file name.")
if loop_counter != 0:
    print(f"Warning: stability file with same name found in \"{abs_directory}\", new filename is \n stability: {filename_stability}")

sp.write_data_flex([a_arr,E_arr, UE_prime_UE_arr, phi_arr, phi_prime_arr, phi_primeprime_arr, Omega_m_arr, Omega_r_arr, Omega_lambda_arr, Omega_phi_arr, M_star_sqrd_arr, alpha_M_arr, alpha_B_arr, alpha_K_arr, w_DE_arr, P_DE_arr, rho_DE_arr],filename_expansion)
sp.write_data_flex([a_arr,chioverdelta_arr,coupling_factor_arr],filename_force)
sp.write_data_flex([a_arr, M_star_sqrd_arr, alpha_M_arr, alpha_B_arr, alpha_K_arr, alpha_M_param1_arr, alpha_B_param1_arr, alpha_K_param1_arr, alpha_M_param2_arr, alpha_B_param2_arr, alpha_K_param2_arr, alpha_M_param3_arr, alpha_B_param3_arr, alpha_K_param3_arr],filename_properties)

if (isinstance(Q_S_arr, np.ndarray) and isinstance(c_s_sq_arr, np.ndarray)):
    sp.write_data_flex([a_arr, Q_S_arr, c_s_sq_arr], filename_stability)
elif isinstance(Q_S_arr, np.ndarray):
    sp.write_data_flex([a_arr, Q_S_arr], filename_stability)
elif isinstance(c_s_sq_arr, np.ndarray):
    sp.write_data_flex([a_arr, c_s_sq_arr], filename_stability)
   
print(f'Files generated. Saved in {abs_directory}')