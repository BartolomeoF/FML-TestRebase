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
Horndeski_path = filenames[0] #Horndeski functional forms and parameter values
numerical_path = filenames[1] #Cosmological and simulation parameter values


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

print('phiprime0 is '+str(phi_prime0))
#---Create Horndeski functions---
###################COMMENT/UNCOMMENT
lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list)
read_out_dict.update(lambdified_functions)
######################################

coupl_fac_sym = sym.latex(sym.simplify(lambdified_functions['coupling_factor_symbolic']))

print('Horndeski functions --------------')
print(f'K = {K}')
print(f'G_3 = {G3}')
print(f'G_4 = {G4}')
print('Coupling Factor-------------------')
print(f'beta = {coupl_fac_sym}')
print('Horndeski parameters--------------')

# #---Run solver and save output---COMMENT/UNCOMMENT
print(model+' model parameters, ' + str(symbol_list)+' = '+str(parameters))
print('Cosmological parameters-----------')
print('Omega_m0 = '+str(Omega_m0))
print('Omega_r0 = '+str(Omega_r0))
print(f'Omega_lambda0 = f{Omega_l0}')
print('Initial conditions----------------')
print(f'Hubble0 = {U0}')
print(f'scalar_prime0 = {phi_prime0}')