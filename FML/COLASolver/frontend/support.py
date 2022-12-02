# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import sys
import matplotlib.pyplot as plt

# writing to file
def write_data(a_arr_inv, E_arr, alpha1_arr, alpha2_arr, B_arr, C_arr, output_filename_as_string):
    datafile_id = open(output_filename_as_string, 'wb')    #here you open the ascii file
    data = np.array([np.array(a_arr_inv[::-1]), np.array(E_arr[::-1]), np.array(alpha1_arr[::-1]), np.array(alpha2_arr[::-1]), np.array(B_arr[::-1]), np.array(C_arr[::-1])])
    data = data.T     #here you transpose your data, so to have it in two columns
    np.savetxt(datafile_id, data, fmt=['%.4e','%.4e','%.4e','%.4e','%.4e','%.4e'])    #here the ascii file is populated.
    datafile_id.close()    #close the file

def write_data_coupl(a_arr_inv, E_arr, B_arr, C_arr, Coupl_arr, output_filename_as_string):
    datafile_id = open(output_filename_as_string, 'wb')    #here you open the ascii file
    data = np.array([np.array(a_arr_inv[::-1]), np.array(E_arr[::-1]), np.array(B_arr[::-1]), np.array(C_arr[::-1]), np.array(Coupl_arr[::-1])])
    data = data.T     #here you transpose your data, so to have it in two columns
    np.savetxt(datafile_id, data, fmt=['%.4e','%.4e','%.4e','%.4e','%.4e'])    #here the ascii file is populated.
    datafile_id.close()    #close the file

def write_data_screencoupl(a_arr_inv, chioverdelta_arr, Coupl_arr, output_filename_as_string):
    datafile_id = open(output_filename_as_string, 'wb')    #here you open the ascii file
    data = np.array([np.array(a_arr_inv[::-1]), np.array(chioverdelta_arr[::-1]), np.array(Coupl_arr[::-1])])
    data = data.T     #here you transpose your data, so to have it in two columns
    np.savetxt(datafile_id, data, fmt=['%.4e','%.4e','%.4e'])    #here the ascii file is populated.
    datafile_id.close()    #close the file

def write_all_data(a_arr_inv, E_arr, E_prime_arr, phi_prime_arr, phi_primeprime_arr, Omega_m_arr, Omega_r_arr, Omega_phi_arr, Omega_L_arr, chioverdelta_arr, coupl_arr, output_filename_as_string):
    datafile_id = open(output_filename_as_string, 'wb')    #here you open the ascii file
    data = np.array([np.array(a_arr_inv[::-1]), np.array(E_arr[::-1]), np.array(E_prime_arr[::-1]), np.array(phi_prime_arr[::-1]), np.array(phi_primeprime_arr[::-1]), np.array(Omega_m_arr[::-1]),np.array(Omega_r_arr[::-1]), np.array(Omega_phi_arr[::-1]), np.array(Omega_L_arr[::-1]),np.array(chioverdelta_arr[::-1]), np.array(coupl_arr[::-1])])
    data = data.T     #here you transpose your data, so to have it in two columns
    np.savetxt(datafile_id, data, fmt=['%.4e','%.4e','%.4e','%.4e','%.4e','%.4e','%.4e','%.4e','%.4e','%.4e','%.4e'])    #here the ascii file is populated.
    datafile_id.close()    #close the file
    
def write_data_flex(data, output_filename_as_string):
    datafile_id = open(output_filename_as_string, 'wb')    #here you open the ascii file
    format_list = list(np.repeat('%.4e',len(data)))
    newdata = []
    for i in data:
        newdata.append(np.array(i[::-1]))
    realdata = np.array(newdata)
    realdata = realdata.T     #here you transpose your data, so to have it in two columns
    np.savetxt(datafile_id, realdata, fmt=format_list)    #here the ascii file is populated.
    datafile_id.close()    #close the file

def make_scan_array(minv, maxv, numv):
    if minv == maxv:
        numv = 1
    return np.linspace(minv,maxv,numv,endpoint=True)

def generate_scan_array(dictionary, quantity_string):
    maxv_string = quantity_string+"_max"
    minv_string = quantity_string+"_min"
    numv_string = quantity_string+"_number"
    maxv = dictionary.as_float(maxv_string)
    minv = dictionary.as_float(minv_string)
    numv = dictionary.as_int(numv_string)
    
    return make_scan_array(minv,maxv,numv)


    
##############################################################################################