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

parser = ArgumentParser(prog='Generate_Simulation_Input')
parser.add_argument('input_ini_filenames',nargs=2)

args = parser.parse_args()
print(args)
filenames = args.input_ini_filenames
Horndeski_path = filenames[0]
numerical_path = filenames[1]

read_out_dict = read_in_parameters(Horndeski_path, numerical_path)
odeint_parameter_symbols = [E, phiprime, omegar, omegam]
read_out_dict.update({'odeint_parameter_symbols':odeint_parameter_symbols})

model = read_out_dict['model_name']
K = read_out_dict['K']
G3 = read_out_dict['G3']
G4 = read_out_dict['G4']
cosmology_name = read_out_dict['cosmo_name']
[Omega_r0, Omega_m0, Omega_l0] = read_out_dict['cosmological_parameters']
[U0, phi_prime0] = read_out_dict['initial_conditions']
[Npoints, z_max, supp_flag, threshold_value, GR_flag] = read_out_dict['simulation_parameters']
parameters = read_out_dict['Horndeski_parameters']
mass_ratio_list = read_out_dict['mass_ratio_list']
symbol_list = read_out_dict['symbol_list']
closure_declaration = read_out_dict['closure_declaration']

print('phiprime0 is '+str(phi_prime0))
#---Create Horndeski functions---
###################COMMENT/UNCOMMENT
lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list)
read_out_dict.update(lambdified_functions)
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

# #---Run solver and save output---COMMENT/UNCOMMENT
print(model+' model parameters, ' + str(symbol_list)+' = '+str(parameters))
print('Cosmological parameters are:')
print('Omega_m0 = '+str(Omega_m0))
print('Omega_r0 = '+str(Omega_r0))
background_quantities = ns.run_solver(read_out_dict)
a_arr = background_quantities['a']
UE_arr = background_quantities['Hubble']
UE_prime_arr = background_quantities['Hubble_prime']
UE_prime_UE_arr = background_quantities['E_prime_E']
coupling_factor_arr = background_quantities['coupling_factor']
chioverdelta_arr = background_quantities['chi_over_delta']

print('Files for Hi-COLA numerical simulation being generated.')
###----Intermediate quantities-----
E_arr = np.array(UE_arr)/U0 #check whether COLA requires intermediates constructed with E rather than U!
E_prime_arr = np.array(UE_prime_arr)/U0 #check whether COLA requires intermediates constructed with Eprime rather than Uprime!
##Note: E_prime_E is the same as U_prime_U, so that array does not need to be multiplied by anything.

directory = '../input/Horndeski'

filename_expansion = directory+f'/{model}_{cosmology_name}_expansion.txt'
filename_force = directory+f'/{model}_{cosmology_name}_force.txt'

print(filename_expansion)

sp.write_data_flex([a_arr,E_arr, UE_prime_UE_arr],filename_expansion)
sp.write_data_flex([a_arr,chioverdelta_arr,coupling_factor_arr],filename_force)
abs_directory = os.path.abspath(directory)
print(f'Files generated. Saved in {abs_directory}')
