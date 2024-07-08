import numpy as np
import sympy as sym
import HiCOLA.Frontend.numerical_solver as ns
import HiCOLA.Frontend.expression_builder as eb
import HiCOLA.Frontend.function_builder as fb
from HiCOLA.Frontend.read_parameters import read_in_parameters
import os
from argparse import ArgumentParser

symbol_decl = eb.declare_symbols()
exec(symbol_decl)

parser = ArgumentParser(prog='Model_Checker')
parser.add_argument('input_ini_filenames',nargs=2)

args = parser.parse_args()
print(args)
filenames = args.input_ini_filenames
Horndeski_path = filenames[0]
numerical_path = filenames[1]

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

lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list, H0)
read_out_dict.update(lambdified_functions)

print(read_out_dict['symbol_list'])
print(read_out_dict['func_list'])
print(read_out_dict['K'])

N_models = 5

for i in range(N_models):
    parameters = fb.generate_params(read_out_dict, N_models)[i]
    read_out_dict.update({'Horndeski_parameters':parameters})

    print(read_out_dict['Horndeski_parameters'])

    background_quantities = ns.run_solver(read_out_dict)

    if background_quantities != False:
        alphas_arr = ns.comp_alphas(read_out_dict, background_quantities)
        background_quantities.update(alphas_arr)

        Q_S_arr, c_s_sq_arr = ns.comp_stability(read_out_dict, background_quantities)