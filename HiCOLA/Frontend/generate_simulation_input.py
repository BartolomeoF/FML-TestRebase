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
H0 = 100*read_out_dict['little_h']

closure_dictionary = {'odeint_parameters':odeint_parameter_symbols, 'parameters':symbol_list}

lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list, H0)
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
if background_quantities == False: 
    print('Warning: The number of elements in some ODE solution(s) is not 1000 due to a numerical discontinuity')
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

LCDM = ns.comp_LCDM(read_out_dict)
E_LCDM_arr = LCDM[0]
E_prime_E_LCDM_arr = LCDM[1]
Omega_m_LCDM_arr = LCDM[2]
Omega_r_LCDM_arr = LCDM[3]
Omega_l_LCDM_arr = LCDM[4]

#----Compute alphas----
alphas_arr = ns.comp_alphas(read_out_dict, background_quantities)
background_quantities.update(alphas_arr)
M_star_sqrd_arr = alphas_arr['M_star_sq']
alpha_M_arr = alphas_arr['alpha_M']
alpha_B_arr = alphas_arr['alpha_B']
alpha_K_arr = alphas_arr['alpha_K']

#----Compute EoS----
DE_arr = ns.comp_w_phi(read_out_dict, background_quantities)#to change compute method remove 2
w_phi_arr = DE_arr[0]
P_phi_arr = DE_arr[1]
rho_phi_arr = DE_arr[2]

w_eff_arr = ns.comp_w_eff(background_quantities)

#----Checking stability conditions----
Q_s_arr, c_s_sq_arr, stable = ns.comp_stability(read_out_dict, background_quantities)
if stable:
    print('Stability conditions satisified')
else:
    print('Warning: Stability conditions not satisfied: Q_s and/or c_s_sq not always > 0')

#----Alpha parameterisation----
z_max = 3

#first parameterisation
#of form alpha_X = alpha_X0*(1 - Omega_m - Omega_r)/(1 - Omega_m0 - Omega_r0)
alpha_X_param1, model_params, fit_goodness = ns.parameterise1(a_arr, z_max, Omega_m0, Omega_r0, alpha_M_arr, alpha_B_arr, alpha_K_arr)
alpha_M_param1_arr, alpha_B_param1_arr, alpha_K_param1_arr = alpha_X_param1
print('First parameterisation------------')
print("alpha_M0, alpha_B0, alpha_K0 = {}, {}, {}".format(*model_params))
print("BIC/N values for alpha_M, alpha_B, alpha_K = {}, {}, {}".format(*fit_goodness))

#second parameterisation
#of form alpha_X = alpha_X0*(1 + z)^(-q)
alpha_X_param2, model_params, fit_goodness = ns.parameterise2(a_arr, z_max, alpha_M_arr, alpha_B_arr, alpha_K_arr)
alpha_M_param2_arr, alpha_B_param2_arr, alpha_K_param2_arr = alpha_X_param2
print('Second parameterisation-----------')
print("alpha_M0, alpha_B0, alpha_K0 = {}, {}, {} \n q = {}".format(*model_params))
print("BIC/N values for alpha_M, alpha_B, alpha_K = {}, {}, {}".format(*fit_goodness))

#third parameterisation
#of form alpha_X = alpha_X0*(1 + z)^(-q_X)
alpha_X_param3, model_params, fit_goodness = ns.parameterise3(a_arr, z_max, alpha_M_arr, alpha_B_arr, alpha_K_arr)
alpha_M_param3_arr, alpha_B_param3_arr, alpha_K_param3_arr = alpha_X_param3
print('Third parameterisation------------')
print("alpha_M0, alpha_B0, alpha_K0 = {}, {}, {} \n q_M, q_B, q_K = {}, {}, {}".format(*model_params))
print("BIC/N values for alpha_M, alpha_B, alpha_K = {}, {}, {}".format(*fit_goodness))

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
filename_LCDM = directory+f'/{model}_{cosmology_name}_LCDM.txt'

abs_directory = os.path.abspath(directory)
loop_counter = 0
while ( os.path.exists(filename_expansion) or os.path.exists(filename_force) or os.path.exists(filename_properties) or os.path.exists(filename_LCDM)) and loop_counter < 100:
    loop_counter += 1
    filename_expansion = sp.renamer(filename_expansion)
    filename_force = sp.renamer(filename_force)
    filename_properties = sp.renamer(filename_properties)
    filename_LCDM = sp.renamer(filename_LCDM)
if loop_counter >= 100:
    raise Exception("Counter for file renaming loop excessively high, consider changing expansion, force and properties output file names.")
if loop_counter != 0:
    print(f"Warning: expansion, force or properties file with same name found in \"{abs_directory}\", new filenames are \n expansion: {filename_expansion} \n force:{filename_force} \n properties:{filename_properties} \n LCDM: {filename_LCDM}")

loop_counter = 0
while os.path.exists(filename_stability) and loop_counter < 100:
    loop_counter += 1
    filename_stability = sp.renamer(filename_stability)
if loop_counter >= 100:
    raise Exception("Counter for file renaming loop excessively high, consider changing stability output file name.")
if loop_counter != 0:
    print(f"Warning: stability file with same name found in \"{abs_directory}\", new filename is \n stability: {filename_stability}")

sp.write_data_flex([a_arr,E_arr, UE_prime_UE_arr, phi_arr, phi_prime_arr, phi_primeprime_arr, Omega_m_arr, Omega_r_arr, Omega_lambda_arr, Omega_phi_arr, M_star_sqrd_arr, alpha_M_arr, alpha_B_arr, alpha_K_arr, w_phi_arr, P_phi_arr, rho_phi_arr, w_eff_arr],filename_expansion)
sp.write_data_flex([a_arr,chioverdelta_arr,coupling_factor_arr],filename_force)
sp.write_data_flex([a_arr, M_star_sqrd_arr, alpha_M_arr, alpha_B_arr, alpha_K_arr, alpha_M_param1_arr, alpha_B_param1_arr, alpha_K_param1_arr, alpha_M_param2_arr, alpha_B_param2_arr, alpha_K_param2_arr, alpha_M_param3_arr, alpha_B_param3_arr, alpha_K_param3_arr],filename_properties)
sp.write_data_flex([a_arr,E_LCDM_arr,E_prime_E_LCDM_arr,Omega_m_LCDM_arr,Omega_r_LCDM_arr,Omega_l_LCDM_arr],filename_LCDM)

if (isinstance(Q_s_arr, np.ndarray) and isinstance(c_s_sq_arr, np.ndarray)):
    sp.write_data_flex([a_arr, Q_s_arr, c_s_sq_arr], filename_stability)
elif isinstance(Q_s_arr, np.ndarray):
    sp.write_data_flex([a_arr, Q_s_arr], filename_stability)
elif isinstance(c_s_sq_arr, np.ndarray):
    sp.write_data_flex([a_arr, c_s_sq_arr], filename_stability)
   
print(f'Files generated. Saved in {abs_directory}')