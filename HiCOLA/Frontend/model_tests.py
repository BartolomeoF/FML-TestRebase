import numpy as np
import HiCOLA.Frontend.numerical_solver as ns

def consistency_check(read_out_dict, background_quantities, tolerance):
    LCDM = ns.comp_LCDM(read_out_dict)
    E_LCDM = LCDM[0]
    E = background_quantities['Hubble']

    consistent = np.max(abs(1-E/E_LCDM))<tolerance
    return consistent