import numpy as np
import sympy as sym
import HiCOLA.Frontend.numerical_solver as ns
import HiCOLA.Frontend.expression_builder as eb
import HiCOLA.Frontend.function_builder as fb
from HiCOLA.Utilities.Other import support as sp
from HiCOLA.Frontend.read_parameters import read_in_parameters
import os
from argparse import ArgumentParser
import time

symbol_decl = eb.declare_symbols()
exec(symbol_decl)

parser = ArgumentParser(prog='Model_Checker')
parser.add_argument('input_ini_filenames', nargs=2)
parser.add_argument('number_of_models', type=int)

args = parser.parse_args()
print(args)
filenames = args.input_ini_filenames
Horndeski_path = filenames[0]
numerical_path = filenames[1]
N_models = args.number_of_models

read_out_dict = read_in_parameters(Horndeski_path, numerical_path)
odeint_parameter_symbols = [E, phi, phiprime, omegar, omegam]
read_out_dict.update({'odeint_parameter_symbols':odeint_parameter_symbols})

reduced = read_out_dict['reduced_flag']

if reduced == True:
    Horndeski_funcs = fb.define_funcs_reduced()
else:
    Horndeski_funcs = fb.define_funcs()
read_out_dict.update(Horndeski_funcs)

K = read_out_dict['K']
G3 = read_out_dict['G3']
G4 = read_out_dict['G4']
mass_ratio_list = read_out_dict['mass_ratio_list']
symbol_list = read_out_dict['symbol_list']
H0 = 100*read_out_dict['little_h']
cosmology_name = read_out_dict['cosmo_name']
directory = read_out_dict['output_directory']

#----producing lambdified functions for all models----
lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list, H0)
read_out_dict.update(lambdified_functions)

#----checking stability of models produced by each set of parameters----
stable_models = []
unstable_models = []
background_list = []
all_parameters = fb.generate_params(read_out_dict, N_models)
any_stable = False

#all_parameters[:, 13:] = 0
#all_parameters[:, 12] = 0.5

start = time.time()
for model_n in range(N_models):
    #print('model {} begin'.format(model_n))
    completion = 100*model_n/N_models
    print('{}% complete'.format(completion), end='\r')
    parameters = all_parameters[model_n]
    read_out_dict.update({'Horndeski_parameters':parameters})

    background_quantities = ns.run_solver(read_out_dict)

    #initial numerical stability check
    if background_quantities == False:
        #print('Warning: The number of steps in some ODE solution(s) is not 1000 due to a numerical discontinuity')
        unstable_models.append(parameters)
    else:
        #stability condition check
        alphas_arr = ns.comp_alphas(read_out_dict, background_quantities)
        background_quantities.update(alphas_arr)

        Q_S_arr, c_s_sq_arr, unstable = ns.comp_stability(read_out_dict, background_quantities)

        if unstable==1:
            #print('Warning: Stability conditions not satisfied: Q_S and c_s_sq not always > 0')
            unstable_models.append(parameters)
        elif unstable==2:
            #print('Warning: Stability condition not satisfied: c_s_sq not always > 0')
            unstable_models.append(parameters)
        elif unstable==3:
            #print('Warning: Stability condition not satisfied: Q_S not always > 0')
            unstable_models.append(parameters)
        # elif (Omega_m_arr<-1e-6).any() or (Omega_r_arr<-1e-6).any() or (Omega_lambda_arr<-1e-6).any() or (Omega_phi_arr<-1e-6).any():
        #     #print('Warning: Background evolution unphysical')
        #     unstable_models.append(parameters)
        else:
            #print('Stability conditions satisified')
            stable_models.append(parameters)

            a_arr = background_quantities['a']
            E_arr = background_quantities['Hubble']  
            phi_arr = background_quantities['scalar']
            phi_prime_arr = background_quantities['scalar_prime']
            Omega_m_arr = background_quantities['omega_m']
            Omega_r_arr = background_quantities['omega_r']
            Omega_lambda_arr = background_quantities['omega_l']
            fried_closure_lambda = read_out_dict['fried_RHS_lambda']
            omega_phi_lambda = read_out_dict['omega_phi_lambda']
            cl_declaration = read_out_dict['closure_declaration']
            [E0, phi0, phi_prime0] = read_out_dict['initial_conditions']

            E_cl_arr = ns.comp_E_closure(fried_closure_lambda, cl_declaration, E0, phi_arr, phi_prime_arr, Omega_r_arr, Omega_m_arr, Omega_lambda_arr, a_arr, parameters)
            Omega_phi_arr = []
            for Ev, phiv, phiprimev, omegalv, omegamv, omegarv in zip(E_cl_arr,phi_arr,phi_prime_arr, Omega_lambda_arr, Omega_m_arr, Omega_r_arr):
                Omega_phi_arr.append(omega_phi_lambda(Ev,phiv,phiprimev,omegalv, omegamv, omegarv,*parameters))

            arr_list = [a_arr,E_arr,phi_arr,phi_prime_arr,Omega_m_arr,Omega_r_arr,Omega_lambda_arr,Omega_phi_arr]
            background_list.append(arr_list)
            any_stable = True
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