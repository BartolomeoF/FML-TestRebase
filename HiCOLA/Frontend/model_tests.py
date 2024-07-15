import numpy as np
import HiCOLA.Frontend.numerical_solver as ns

def try_solver(run_solver, read_out_dict):
    """
    This function attempts to run a numerical solver using the provided run_solver function and read_out_dic as input.

    Parameters:
    run_solver (func): Represents the numerical solver to be run that takes read_out_dict as input.
    read_out_dict (dict): Contains input parameters for the numerical solver.

    Returns:
    background_quantities (dict): Contains the background quantities obtained from the numerical solver. If an 
    exception occurs during the solver execution, the function will return False.

    Raises:
    Exception: If a timeout event occurs during the solver execution.
    """
    try:
        background_quantities = run_solver(read_out_dict)
    except Exception as ex:
        print('Exception: ({}) occurred due to timeout event in solver.'.format(ex))
        background_quantities = False
    finally:
        return background_quantities

def consistency_check(read_out_dict, background_quantities, tolerance):
    """
    This function checks the consistency of the background quantities obtained from the numerical solver with the LCDM model.

    Parameters:
    read_out_dict (dict): Contains input parameters for the numerical solver.
    background_quantities (dict): Contains the background quantities obtained from the numerical solver.
    tolerance (float): Defines the maximum allowed difference between the background quantities and the LCDM model.

    Returns:
    consistent (bool): Indicates whether the background quantities are consistent with the LCDM model within specified tolerance.
    """
    LCDM = ns.comp_LCDM(read_out_dict)
    E_LCDM = LCDM[0]
    E = background_quantities['Hubble']

    consistent = np.max(abs(1-E/E_LCDM))<tolerance
    return consistent

def check_background(background_quantities):
    """
    This function checks if the background quantities obtained from the numerical solver have positive values for Omega_m, Omega_r, Omega_l, and Omega_phi.

    Parameters:
    background_quantities (dict): Contains the background quantities obtained from the numerical solver.

    Returns:
    bool: Returns True if all the density parameters are positive, otherwise returns False.
    """
    Omega_m = background_quantities['omega_m']
    Omega_r = background_quantities['omega_r']
    Omega_l = background_quantities['omega_l']
    Omega_phi = background_quantities['omega_phi']
    if (Omega_m>0).all() and (Omega_r>0).all() and (Omega_l>0).all() and (Omega_phi>0).all():
        return True
    else:
        return False