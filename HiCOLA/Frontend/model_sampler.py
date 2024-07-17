import numpy as np
import sympy as sym
import HiCOLA.Frontend.numerical_solver as ns
import HiCOLA.Frontend.expression_builder as eb
import HiCOLA.Frontend.function_builder as fb
import HiCOLA.Frontend.model_tests as mt
from HiCOLA.Utilities.Other import support as sp
from HiCOLA.Frontend.read_parameters import read_in_parameters
import os
from argparse import ArgumentParser
import time

symbol_decl = eb.declare_symbols()
exec(symbol_decl)

#command line parameters
parser = ArgumentParser(prog='Model_Sampler')
parser.add_argument('input_ini_filenames', nargs=2)

args = parser.parse_args()
print(args)
filenames = args.input_ini_filenames
Horndeski_path = filenames[0]
numerical_path = filenames[1]

#----general set up for all models----
#reading in parameters from files
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
[Npoints, z_max, suppression_flag, threshold, GR_flag] = read_out_dict['simulation_parameters']
[Omega_r0, Omega_m0, Omega_l0] = read_out_dict['cosmological_parameters']

#producing lambdified functions
lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list, H0)
read_out_dict.update(lambdified_functions)

z_final = 0.
x_ini = np.log(1./(1.+z_max))
x_final = np.log(1./(1.+z_final))
x_arr = np.linspace(x_ini, x_final, Npoints)
a_arr = [np.exp(x) for x in x_arr]
a_arr = np.array(a_arr)
z_arr = 1/a_arr - 1
E_LCDM = ns.comp_E_LCDM(z_arr, Omega_r0, Omega_m0)

E_err = 0.2
data = (read_out_dict, E_LCDM, E_err)
nwalkers = 50
niter = 200
initial = np.array([-0.00078507, -0.01592667, -0.01252403, -0.00750742, -0.00913699]) #if using final value of last chain don't need burn in
dim = len(initial)
rng = np.random.default_rng()
p0 = [np.array(initial) + 1e-7 * rng.standard_normal(dim) for i in range(nwalkers)]

probability = fb.create_prob_glob(read_out_dict, E_LCDM, E_err)

sampler, pos, prob, state = fb.main(p0, nwalkers, niter, dim, probability)

#fb.plotter(sampler, read_out_dict, z_arr, E_LCDM)

samples = sampler.flatchain
probs = sampler.flatlnprobability

filename_samples = directory+f'/samples.txt'
filename_probs = directory+f'/probabilities.txt'

abs_directory = os.path.abspath(directory)
loop_counter = 0
while (os.path.exists(filename_samples) or os.path.exists(filename_probs)) and loop_counter < 100:
    loop_counter += 1
    filename_samples = sp.renamer(filename_samples)
    filename_probs = sp.renamer(filename_probs)
if loop_counter >= 100:
    raise Exception("Counter for file renaming loop excessively high, consider changing samples and probs output file names.")
if loop_counter != 0:
    print(f"Warning: samples or probs file with same name found in \"{abs_directory}\", new filenames are \n samples: {filename_samples} \n probs:{filename_probs}")

sp.write_model_list(samples, filename_samples)
sp.write_data_flex([probs], filename_probs)