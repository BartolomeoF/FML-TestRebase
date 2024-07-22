import numpy as np
import sympy as sym
import HiCOLA.Frontend.numerical_solver as ns
import HiCOLA.Utilities.ModelEvaluation.model_tests as mt
import emcee
import matplotlib.pyplot as plt

def symloguniform(size, low=-13.0, high=13.0):
    """
    Generate random samples from a symmetric logarithmic uniform distribution. Whole distribution ranges from -1 to 1 but is split into positive and 
    negative components with intervals of (exp(low), 1) and (-1, -exp(-high)), meaning no values in the interval (-exp(-high), exp(low)) can be generated.

    Parameters:
    size (int or tuple of ints): Output shape.
    low (float, optional): The lower bound of the distribution for the positive outputs. Default is -13.0 corresponding to a lowest value of ~2e-6.
    high (float, optional): The upper bound of the distribution for the negative outputs. Default is 13.0 corresponding to a highest value of ~-2e-6.

    Returns:
    symlog (ndarray): A numpy array containing the random samples from the symmetric logarithmic uniform distribution.
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
    out (dict): Contains the defined Horndeski functions and their corresponding symbols.
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
    out (dict): Contains the defined Horndeski functions and their corresponding symbols.
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
    read_out_dict (dict): Contains the Horndeski functions and their corresponding symbols.
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
    """
    This function computes the Hubble parameter E for a given set of Horndeski parameters.
    It also checks the stability conditions of the model by computing the alphas and comparing them to the stability criteria.

    Parameters:
    theta (tuple): Contains the Horndeski parameters being sampled.
    read_out_dict (dict): Contains the Horndeski functions and their corresponding symbols.

    Returns:
    E (ndarray): The computed Hubble parameter E and a 
    unstable (bool): Indicates whether the model is unstable (True) or not (False).
    """
    k_phi, k_X, g_3phi, g_3X, g_4phi = theta

    parameters = k_phi, k_X, g_3phi, g_3X, g_4phi
    read_out_dict.update({'Horndeski_parameters':parameters})
    background_quantities = mt.try_solver(ns.run_solver_lite, read_out_dict)
    E = background_quantities['Hubble']

    if not isinstance(background_quantities['scalar'], bool):#i.e. if there is no numerical discontinuity
        #computing whether stability conditions are satisfied
        alphas_arr = ns.comp_alphas(read_out_dict, background_quantities)
        background_quantities.update(alphas_arr)
        unstable = ns.comp_stability(read_out_dict, background_quantities)[2]
        return E, unstable
    unstable = True
    return E, unstable

def log_likelihood(theta, read_out_dict, E, E_err):
    """
    Parameters:
    theta (tuple): Contains the Horndeski parameters being sampled.
    read_out_dict (dict): Contains the Horndeski functions and their corresponding symbols.
    E (float): The Hubble parameter being compared to.
    E_err (float): The weighting on the model values of E.

    Returns:
    float: The computed log-likelihood value.
    """
    mod_E, unstable = model_E(theta, read_out_dict)
    if unstable:
        return -np.inf
    return -0.5*np.sum(((1-E/mod_E)/E_err)**2)

def log_prior(theta):
    """
    Parameters:
    theta (tuple): Contains the Horndeski parameters being sampled.

    Returns:
    float: If the parameters are within the allowed range, the function returns 0.0. 
    If any of the parameters are outside the allowed range, the function returns -np.inf.
    """
    k_phi, k_X, g_3phi, g_3X, g_4phi = theta
    if -1 < k_phi < 1 and -1 < k_X < 1 and -1 < g_3phi < 1 and -1 < g_3X < 1 and -1 < g_4phi < 1:
        return 0.0
    return -np.inf

def create_prob_glob(read_out_dict, E, E_err):
    """
    Parameters:
    read_out_dict (dict): Contains the Horndeski functions and their corresponding symbols.
    E (float): The Hubble parameter being compared to.
    E_err (float): The weighting on the model values of E.

    Returns:
    log_probability_global (function): A function that combines the results of log_likelihood and log_prior for a given set of Horndeski parameters.
    """
    def log_probability_global(theta):
        p = log_prior(theta)
        if not np.isfinite(p):
            return -np.inf
        return p + log_likelihood(theta, read_out_dict, E, E_err)
    return log_probability_global

def main(p0, nwalkers, niter, dim, probability, noburnin):
    """
    Parameters:
    p0 (ndarray): Initial guess for the parameters.
    nwalkers (int): Number of random walkers.
    niter (int): Number of iterations.
    dim (int): Dimensionality of the parameter space.
    probability (function): A function that combines the results of log_likelihood and priors for a given set of Horndeski parameters.
    noburnin (bool): Indicates whether to run a burn-in period (False) or not (True).

    Returns:
    sampler (emcee.EnsembleSampler): An instance of the EnsembleSampler class from the emcee package.
    pos (ndarray): The positions of the walkers.
    prob (ndarray): The log-probabilities of the walkers.
    state (dict): The state dictionary containing the current state of the MCMC sampling.
    """
    sampler = emcee.EnsembleSampler(nwalkers, dim, probability)

    if noburnin==False:
        print('Running burn-in...')
        p0, _, _ = sampler.run_mcmc(p0, 100, progress = True)
        sampler.reset()

    print('Running production...')
    pos, prob, state = sampler.run_mcmc(p0, niter, progress = True)

    return sampler, pos, prob, state

def sample_walkers(nsamples, flatchain, read_out_dict):
    """
    This function samples a specified number of walkers from a given flattened chain.

    Parameters:
    nsamples (int): The number of walkers to sample from the chain.
    flatchain (np.ndarray): A 2D numpy array containing the random parameter values for each model.
    read_out_dict (dict): Contains the Horndeski functions and their corresponding symbols.

    Returns:
    med_model (np.ndarray): A 1D numpy array containing the median values of the modelled Hubble parameter E for the sampled walkers.
    spread (np.ndarray): A 1D numpy array containing the standard deviation values of the modelled Hubble parameter E for the sampled walkers.
    """
    models = []
    rng = np.random.default_rng()
    draw = rng.integers(0,len(flatchain),size=nsamples)
    thetas = flatchain[draw]
    for i in thetas:
        mod = model_E(i, read_out_dict)[0]
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model,spread
