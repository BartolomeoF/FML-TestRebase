import numpy as np
import sympy as sym

def define_funcs():
    K, G3, G4, k, k_phi, k_X, k_phiphi, k_phiX, k_XX, g_3, g_3phi, g_3X, g_3phiphi, g_3phiX, g_3XX, g_4, g_4phi, g_4phiphi, phi, X, n, m = sym.symbols(r'K G_{3} G_{4} k k_{\phi} k_{X} k_{\phi\phi} k_{X\phi} k_{XX} g_{3} g_{3\phi} g_{3X} g_{3\phi\phi} g_{3X\phi} g_{3XX} g_{4} g_{4\phi} g_{4\phi\phi} phi X n m')

    K = k + k_phi*phi + k_X*X + k_phiphi*phi*phi + k_phiX*phi*X + k_XX*X*X
    G3 = g_3 + g_3phi*phi + g_3X*X + g_3phiphi*phi*phi + g_3phiX*phi*X + g_3XX*X*X
    G4 = g_4 + g_4phi*phi + g_4phiphi*phi*phi

    Horndeski_functions = {'K':K, 'G3':G3, 'G4':G4}
    symbol_list = [k, k_phi, k_X, k_phiphi, k_phiX, k_XX,
                    g_3, g_3phi, g_3X, g_3phiphi, g_3phiX, g_3XX,
                    g_4, g_4phi, g_4phiphi]
    func_list = [K, G3, G4]
    out = {'symbol_list':symbol_list, 'func_list':func_list}
    out.update(Horndeski_functions)
    return out

def generate_params(read_out_dict, N_models):
    """
    Generate random parameter values for a list of symbolic functions.

    Parameters:
    funcs (list): A list of symbolic functions. Each function should be a sympy expression.

    Returns:
    numpy.ndarray: A 1D numpy array containing the random parameter values.

    The function iterates over each function in the input list. It extracts the symbols 
    (parameters) using the sympy atoms() method. phi and X are variables and so no values 
    are generated for those symbols.
    """
    rng = np.random.default_rng()
    parameters_tot = set()
    phi, X = sym.symbols('phi X')

    funcs = read_out_dict['func_list']

    for func in funcs:
        parameters = func.atoms(sym.Symbol)
        if (any(x == phi for x in parameters) and any(x == X for x in parameters)):
            parameters.remove(phi)
            parameters.remove(X)
        else:
            parameters.remove(phi)
        parameters_tot.update(parameters)

    param_vals = rng.uniform(size=(N_models, len(parameters_tot)))
    return param_vals