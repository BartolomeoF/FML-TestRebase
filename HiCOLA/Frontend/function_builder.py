import numpy as np
import sympy as sym
import HiCOLA.Frontend.numerical_solver as ns
import HiCOLA.Frontend.model_tests as mt
import emcee
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool

def symloguniform(size, low=-13.0, high=13.0):
    """
    Generate random samples from a symmetric logarithmic uniform distribution. Whole distribution ranges from -1 to 1 but is split into positive and 
    negative components with intervals of (exp(low), 1) and (-1, -exp(-high)), meaning no values in the interval (-exp(-high), exp(low)) can be generated.

    Parameters:
    size (int or tuple of ints): Output shape.
    low (float, optional): The lower bound of the distribution for the positive outputs. Default is -13.0 corresponding to a lowest value of ~2e-6.
    high (float, optional): The upper bound of the distribution for the negative outputs. Default is 13.0 corresponding to a highest value of ~-2e-6.

    Returns:
    ndarray: A numpy array containing the random samples from the symmetric logarithmic uniform distribution.
    """
    x = np.random.uniform(low, high, size)
    symlog = x.copy()
    symlog[x<0] = np.exp(x[x<0])
    symlog[x>0] = -np.exp(-x[x>0])
    return symlog

def define_funcs():
    """
    This function defines the Horndeski functions K, G3, and G4 using 15 free parameters.

    Returns:
    out (dict): A dictionary containing the defined Horndeski functions and their corresponding symbols.
    """
    K, G3, G4, k, k_phi, k_X, k_phiphi, k_phiX, k_XX, g_3, g_3phi, g_3X, g_3phiphi, g_3phiX, g_3XX, g_4, g_4phi, g_4phiphi, phi, X, n, m = sym.symbols(r'K G_{3} G_{4} k k_{\phi} k_{X} k_{\phi\phi} k_{X\phi} k_{XX} g_{3} g_{3\phi} g_{3X} g_{3\phi\phi} g_{3X\phi} g_{3XX} g_{4} g_{4\phi} g_{4\phi\phi} phi X n m')

    K = k + k_phi*phi + k_X*X + k_phiphi*phi*phi + k_phiX*phi*X + k_XX*X*X
    G3 = g_3 + g_3phi*phi + g_3X*X + g_3phiphi*phi*phi + g_3phiX*phi*X + g_3XX*X*X
    G4 = g_4 + g_4phi*phi + g_4phiphi*phi*phi

    Horndeski_functions = {'K':K, 'G3':G3, 'G4':G4} #to overwrite existing funcs produced by dummy parameters
    symbol_list = [k, k_phi, k_X, k_phiphi, k_phiX, k_XX,
                    g_3, g_3phi, g_3X, g_3phiphi, g_3phiX, g_3XX,
                    g_4, g_4phi, g_4phiphi]
    func_list = [K, G3, G4]
    out = {'symbol_list':symbol_list, 'func_list':func_list} #func_list used by generate_params
    out.update(Horndeski_functions)
    return out

def define_funcs_reduced():
    """
    This function defines the Horndeski functions K, G3, and G4 using 5 free parameters.

    Returns:
    out (dict): A dictionary containing the defined Horndeski functions and their corresponding symbols.
    """
    K, G3, G4, k_phi, k_X, g_3phi, g_3X, g_4phi, phi, X, n, m = sym.symbols(r'K G_{3} G_{4} k_{\phi} k_{X} g_{3\phi} g_{3X} g_{4\phi} phi X n m')

    K = k_phi*phi + k_X*X
    G3 = g_3phi*phi + g_3X*X
    G4 = 0.5 + g_4phi*phi

    Horndeski_functions = {'K':K, 'G3':G3, 'G4':G4} #to overwrite existing funcs produced by dummy parameters
    symbol_list = [k_phi, k_X,
                    g_3phi, g_3X,
                    g_4phi]
    func_list = [K, G3, G4]
    out = {'symbol_list':symbol_list, 'func_list':func_list} #func_list used by generate_params
    out.update(Horndeski_functions)
    return out

def generate_params(read_out_dict, N_models):
    """
    This function generates random parameter values for the given Horndeski functions.

    Parameters:
    read_out_dict (dict): A dictionary containing the Horndeski functions and their corresponding symbols.
    N_models (int): The number of random parameter sets to generate.

    Returns:
    param_vals (ndarray): A 2D numpy array containing the random parameter values for each model.
    """
    rng = np.random.default_rng()
    parameters_tot = set()
    phi, X = sym.symbols('phi X')

    funcs = read_out_dict['func_list']

    for func in funcs:
        parameters = func.atoms(sym.Symbol) #extracting the list of parameter symbols
        if (any(x == phi for x in parameters) and any(x == X for x in parameters)):
            parameters.remove(phi)
            parameters.remove(X)
        else:
            parameters.remove(phi)
        parameters_tot.update(parameters)

    param_vals = symloguniform((N_models, len(parameters_tot))) #generating 'N_models' random sets of parameters
    return param_vals

def model_E(theta, read_out_dict):
    k_phi, k_X, g_3phi, g_3X, g_4phi = theta

    parameters = k_phi, k_X, g_3phi, g_3X, g_4phi
    read_out_dict.update({'Horndeski_parameters':parameters})
    background_quantities = mt.try_solver(ns.run_solver, read_out_dict)
    E = background_quantities['Hubble']
    return E

def log_likelihood(theta, read_out_dict, E, E_err):
    return -0.5*np.sum(((1-E/model_E(theta, read_out_dict))/E_err)**2)

def prior(theta):
    k_phi, k_X, g_3phi, g_3X, g_4phi = theta
    if -1 < k_phi < 1 and -1 < k_X < 1 and -1 < g_3phi < 1 and -1 < g_3X < 1 and -1 < g_4phi < 1:
        return 0.0
    return -np.inf

def probability(theta, read_out_dict, E, E_err):
    p = prior(theta)
    if not np.isfinite(p):
        return -np.inf
    return p + log_likelihood(theta, read_out_dict, E, E_err)

def create_prob_glob(read_out_dict, E, E_err):
    def probability_global(theta):
        p = prior(theta)
        if not np.isfinite(p):
            return -np.inf
        return p + log_likelihood(theta, read_out_dict, E, E_err)
    return probability_global

def main(p0, nwalkers, niter, dim, probability, data):
    #with ProcessingPool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, dim, probability, args=data)

    # print('Running burn-in...')
    # p0, _, _ = sampler.run_mcmc(p0, 100, progress = True)
    # sampler.reset()

    print('Running production...')
    pos, prob, state = sampler.run_mcmc(p0, niter, progress = True)

    return sampler, pos, prob, state

def plotter(sampler, read_out_dict, z, E):
    print('Creating figure...')
    rng = np.random.default_rng()
    samples = sampler.flatchain
    for theta in samples:
        plt.plot(z, model_E(theta, read_out_dict)/E, color='r', alpha=0.1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel('z')
    plt.ylabel('E/E_LCDM')
    plt.xscale('log')
    plt.savefig('MCMC.png')
