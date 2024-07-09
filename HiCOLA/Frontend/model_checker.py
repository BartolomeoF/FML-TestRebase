import numpy as np
import sympy as sym
import HiCOLA.Frontend.numerical_solver as ns
import HiCOLA.Frontend.expression_builder as eb
import HiCOLA.Frontend.function_builder as fb
from HiCOLA.Utilities.Other import support as sp
from HiCOLA.Frontend.read_parameters import read_in_parameters
import os
from argparse import ArgumentParser

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

lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list, H0)
read_out_dict.update(lambdified_functions)

print(read_out_dict['symbol_list'])
print(read_out_dict['func_list'])
print(read_out_dict['K'])

stable_models = []
unstable_models = []

for i in range(N_models):
    parameters = fb.generate_params(read_out_dict, N_models)[i]
    read_out_dict.update({'Horndeski_parameters':parameters})

    background_quantities = ns.run_solver(read_out_dict)

    if background_quantities == False:
        #print('Warning: The number of steps in some ODE solution(s) is not 1000 due to a numerical discontinuity')
        unstable_models.append(read_out_dict['Horndeski_parameters'])
    else:
        alphas_arr = ns.comp_alphas(read_out_dict, background_quantities)
        background_quantities.update(alphas_arr)

        Q_S_arr, c_s_sq_arr, stable = ns.comp_stability(read_out_dict, background_quantities)
        if stable:
            #print('Stability conditions satisified')
            stable_models.append(read_out_dict['Horndeski_parameters'])
        else:
            #print('Warning: Stability conditions not satisfied: Q_S and/or c_s_sq not always > 0')
            unstable_models.append(read_out_dict['Horndeski_parameters'])

filename_stable = directory+f'/{cosmology_name}_stable.txt'
filename_unstable = directory+f'/{cosmology_name}_unstable.txt'

abs_directory = os.path.abspath(directory)
loop_counter = 0
while ( os.path.exists(filename_stable) or os.path.exists(filename_unstable)) and loop_counter < 100:
    loop_counter += 1
    filename_stable = sp.renamer(filename_stable)
    filename_unstable = sp.renamer(filename_unstable)
if loop_counter >= 100:
    raise Exception("Counter for file renaming loop excessively high, consider changing stable and unstable output file names.")
if loop_counter != 0:
    print(f"Warning: stable or unstable file with same name found in \"{abs_directory}\", new filenames are \n stable: {filename_stable} \n unstable:{filename_unstable}")

sp.write_model_list(stable_models, filename_stable)
sp.write_model_list(unstable_models, filename_unstable)