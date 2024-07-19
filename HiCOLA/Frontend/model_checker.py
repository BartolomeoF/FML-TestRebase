import numpy as np
import sympy as sym
import HiCOLA.Frontend.numerical_solver as ns
import HiCOLA.Frontend.expression_builder as eb
import HiCOLA.Frontend.model_builder as mb
import HiCOLA.Frontend.model_tests as mt
from HiCOLA.Utilities.Other import support as sp
from HiCOLA.Frontend.read_parameters import read_in_parameters
import os
from argparse import ArgumentParser
import time
from tqdm import tqdm

symbol_decl = eb.declare_symbols()
exec(symbol_decl)

#command line parameters
parser = ArgumentParser(prog='Model_Checker')
parser.add_argument('input_ini_filenames', nargs=2)
parser.add_argument('number_of_models', type=int)

args = parser.parse_args()
print(args)
filenames = args.input_ini_filenames
Horndeski_path = filenames[0]
numerical_path = filenames[1]
N_models = args.number_of_models

#----general set up for all models----
#reading in parameters from files
read_out_dict = read_in_parameters(Horndeski_path, numerical_path)
odeint_parameter_symbols = [E, phi, phiprime, omegar, omegam]
read_out_dict.update({'odeint_parameter_symbols':odeint_parameter_symbols})

reduced = read_out_dict['reduced_flag']
if reduced == True:
    Horndeski_funcs = mb.define_funcs_reduced()
else:
    Horndeski_funcs = mb.define_funcs()
read_out_dict.update(Horndeski_funcs)

K = read_out_dict['K']
G3 = read_out_dict['G3']
G4 = read_out_dict['G4']
mass_ratio_list = read_out_dict['mass_ratio_list']
symbol_list = read_out_dict['symbol_list']
H0 = 100*read_out_dict['little_h']
cosmology_name = read_out_dict['cosmo_name']
directory = read_out_dict['output_directory']

#producing lambdified functions
lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list, H0)
read_out_dict.update(lambdified_functions)

#generating random parameters
all_parameters = mb.generate_params(read_out_dict, N_models)

#----checking consistency and stability of models produced by each set of parameters----
stable_models = []
unstable_models = []
background_list = []
any_stable = False

start = time.time()
for model_n in range(N_models):
    #print('model {} begin'.format(model_n))
    completion = 100*model_n/N_models
    print('{}% complete'.format(completion), end='\r')
    parameters = all_parameters[model_n]
    read_out_dict.update({'Horndeski_parameters':parameters})

    background_quantities = mt.try_solver(ns.run_solver, read_out_dict)

    #initial numerical stability check
    if isinstance(background_quantities['scalar'], bool): #scalar is set to False if a numerical discontinuity occurs
        #print('Warning: The number of steps in some ODE solution(s) is not 1000 due to a numerical discontinuity')
        unstable_models.append(parameters)
    else:
        #consistency check
        consistent = mt.consistency_check(read_out_dict, background_quantities, 0.2)

        if consistent:
            #print('Consistent')
            alphas_arr = ns.comp_alphas(read_out_dict, background_quantities)
            background_quantities.update(alphas_arr)

            #stability condition check
            Q_s_arr, c_s_sq_arr, unstable = ns.comp_stability(read_out_dict, background_quantities)

            if unstable==1:
                #print('Warning: Stability conditions not satisfied: Q_s and c_s_sq not always > 0')
                unstable_models.append(parameters)
            elif unstable==2:
                #print('Warning: Stability condition not satisfied: c_s_sq not always > 0')
                unstable_models.append(parameters)
            elif unstable==3:
                #print('Warning: Stability condition not satisfied: Q_s not always > 0')
                unstable_models.append(parameters)
            else:
                #print('Stability conditions satisfied')
                stable_models.append(parameters)

                a_arr = background_quantities['a']
                E_arr = background_quantities['Hubble']  
                phi_arr = background_quantities['scalar']
                phi_prime_arr = background_quantities['scalar_prime']
                Omega_m_arr = background_quantities['omega_m']
                Omega_r_arr = background_quantities['omega_r']
                Omega_lambda_arr = background_quantities['omega_l']
                Omega_phi_arr = background_quantities['omega_phi']

                arr_list = [a_arr,E_arr,phi_arr,phi_prime_arr,Omega_m_arr,Omega_r_arr,Omega_lambda_arr,Omega_phi_arr]
                background_list.append(arr_list)
                any_stable = True
        else:
            unstable_models.append(parameters)
end = time.time()

#----writing stable and unstable models to files----
filename_stable = directory+f'/{cosmology_name}_stable.txt'
filename_unstable = directory+f'/{cosmology_name}_unstable.txt'
filename_expansion = directory+f'/{cosmology_name}_expansion.txt'

abs_directory = os.path.abspath(directory)
loop_counter = 0
while ( os.path.exists(filename_stable) or os.path.exists(filename_unstable) or os.path.exists(filename_expansion)) and loop_counter < 100:
    loop_counter += 1
    filename_stable = sp.renamer(filename_stable)
    filename_unstable = sp.renamer(filename_unstable)
    filename_expansion = sp.renamer(filename_expansion)
if loop_counter >= 100:
    raise Exception("Counter for file renaming loop excessively high, consider changing stable and unstable output file names.")
if loop_counter != 0:
    print(f"Warning: stable or unstable or expansion file with same name found in \"{abs_directory}\", new filenames are \n stable: {filename_stable} \n unstable:{filename_unstable} \n expansion:{filename_expansion}")

sp.write_model_list(stable_models, filename_stable)
sp.write_model_list(unstable_models, filename_unstable)
if any_stable:
    sp.write_files(background_list, filename_expansion)

print('{} stable and {} unstable models out of {}'.format(len(stable_models), len(unstable_models), N_models))
print('time taken = {}s'.format(end-start))