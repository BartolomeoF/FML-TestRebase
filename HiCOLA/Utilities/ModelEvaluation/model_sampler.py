import numpy as np
import sympy as sym
import HiCOLA.Frontend.numerical_solver as ns
import HiCOLA.Frontend.expression_builder as eb
import HiCOLA.Utilities.ModelEvaluation.model_builder as mb
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
parser.add_argument('number_of_walkers', type=int)
parser.add_argument('number_of_iterations', type=int)
parser.add_argument('--no_burn_in', action='store_true', default=False)

args = parser.parse_args()
print(args)
filenames = args.input_ini_filenames
Horndeski_path = filenames[0]
numerical_path = filenames[1]
nwalkers = args.number_of_walkers
niter = args.number_of_iterations
noburnin = args.no_burn_in

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
[Npoints, z_max, suppression_flag, threshold, GR_flag] = read_out_dict['simulation_parameters']
[Omega_r0, Omega_m0, Omega_l0] = read_out_dict['cosmological_parameters']

#producing lambdified functions
lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list, H0)
read_out_dict.update(lambdified_functions)

#setting z values
z_final = 0.
x_ini = np.log(1./(1.+z_max))
x_final = np.log(1./(1.+z_final))
x_arr = np.linspace(x_ini, x_final, Npoints)
a_arr = [np.exp(x) for x in x_arr]
a_arr = np.array(a_arr)
z_arr = 1/a_arr - 1

E_LCDM = ns.comp_E_LCDM(z_arr, Omega_r0, Omega_m0)

#----sampling parameter space with MCMC----
#setting MCMC parameters
E_err = 0.15/(0.5 + np.exp(-0.001*z_arr**3))
data = (read_out_dict, E_LCDM, E_err)
initial = np.array([0.0, 0.0, 0.0 ,0.0, 0.0]) #if using final value of last chain don't need burn in
dim = len(initial)
rng = np.random.default_rng()
p0 = [np.array(initial) + 1e-7 * rng.standard_normal(dim) for i in range(nwalkers)]
log_probability = mb.create_prob_glob(read_out_dict, E_LCDM, E_err)

#running MCMC
sampler, pos, prob, state = mb.main(p0, nwalkers, niter, dim, log_probability, noburnin)

samples = sampler.flatchain
probs = sampler.flatlnprobability

#finding 25 distinct most likely models
descending = np.sort(np.unique(probs))[::-1]
indices = [] #fining indices of unique sorted probs
for value in descending[:25]:
    index = np.where(probs == value)[0][0]
    indices.append(index)
theta_max = samples[indices] #parameter values corresponding to indices
best_fit_Es = []
best_w_phis = []
#computing E and w_phi for selected parameter vals
for theta in theta_max:
    model = mb.model_E(theta, read_out_dict)
    best_fit_Es.append(model[0])
    best_w_phis.append(model[2])
best_fit_E = best_fit_Es[0]

#finding posterior spread
med_model, spread = mb.sample_walkers(100, samples, read_out_dict)

#----writing samples and probabilities to files----
filename_samples = directory+f'/samples.txt'
filename_probs = directory+f'/probabilities.txt'
filename_posterior = directory+f'/posterior-input.txt'
filename_best = directory+f'/best-models.txt'

abs_directory = os.path.abspath(directory)
loop_counter = 0
while (os.path.exists(filename_samples) or os.path.exists(filename_probs) or os.path.exists(filename_posterior) or os.path.exists(filename_best)) and loop_counter < 100:
    loop_counter += 1
    filename_samples = sp.renamer(filename_samples)
    filename_probs = sp.renamer(filename_probs)
    filename_posterior = sp.renamer(filename_posterior)
    filename_best = sp.renamer(filename_best)
if loop_counter >= 100:
    raise Exception("Counter for file renaming loop excessively high, consider changing samples and probs output file names.")
if loop_counter != 0:
    print(f"Warning: samples or probs file with same name found in \"{abs_directory}\", new filenames are \n samples: {filename_samples} \n probs:{filename_probs} \n posterior-input: {filename_posterior} \n best-models: {filename_best}")

sp.write_model_list(samples, filename_samples)
sp.write_data_flex([probs[::-1]], filename_probs) #writes reverse
sp.write_data_flex([z_arr, E_LCDM, best_fit_E, med_model, spread], filename_posterior)
sp.write_data_flex(best_fit_Es+best_w_phis, filename_best)