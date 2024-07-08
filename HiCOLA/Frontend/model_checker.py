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

parameters = fb.generate_params(read_out_dict)
read_out_dict.update(parameters)

print(read_out_dict['symbol_list'])
print(read_out_dict['func_list'])
print(read_out_dict['K'])
print(read_out_dict['Horndeski_parameters'])