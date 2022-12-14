# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import sys
import matplotlib.pyplot as plt
import itertools as it

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

def ESS_dS_parameters(EdS, f, k1seed, g31seed,Omega_r0h2 = 4.28e-5, Omega_b0h2 = 0.02196, Omega_c0h2 = 0.1274, h = 0.7307):
    Omega_r0 = Omega_r0h2/h/h
    Omega_m0 = (Omega_b0h2 + Omega_c0h2)/h/h
    Omega_DE0 = 1. - Omega_r0 - Omega_m0
    Omega_l0 = (1.-f)*Omega_DE0
    
    U0 = 1./EdS
    
    alpha_expr = 1.-Omega_l0/EdS/EdS

    k1_dS = k1seed*alpha_expr         #k1_dSv has been called k1(dS)-seed in my notes
    k2_dS = -2.*k1_dS - 12.*alpha_expr
    g31_dS = g31seed*alpha_expr#2.*alpha_expr                 #g31_dSv is g31(dS)-seed
    g32_dS = 0.5*( 1.- ( Omega_l0/EdS/EdS + k1_dS/6. + k2_dS/4. + g31_dS) )
    parameters = [k1_dS,k2_dS, g31_dS, g32_dS]
    return U0, Omega_r0, Omega_m0, Omega_l0, parameters

def ESS_seed_to_direct_scanning_values(odeint_values_scan_filename, parameter_values_scan_filename, EdS_range, f_range, k1seed_range, g31seed_range, Omega_r0h2 = 4.28e-5, Omega_b0h2 = 0.02196, Omega_c0h2 = 0.1274, h = 0.7307, phiprime0 = 0.9):
    

    EdS_array = make_scan_array(*EdS_range)
    f_array = make_scan_array(*f_range)
    k1seed_array = make_scan_array(*k1seed_range)
    g31seed_array = make_scan_array(*g31seed_range)
    
    seed_cart_prod = it.product(EdS_array, f_array, k1seed_array, g31seed_array)
    seed_cart_prod2 = it.product(EdS_array, f_array, k1seed_array, g31seed_array)
    print(len(list(seed_cart_prod2)))
    U0_list = []
    phi_prime0_list = []
    Omega_m0_list = []
    Omega_r0_list = []
    Omega_l0_list = []
    k1_list = []
    k2_list = []
    g31_list = []
    g32_list = []
    for i in seed_cart_prod:
        EdS, f, k1seed, g31seed = i
        U0, Omega_r0, Omega_m0, Omega_l0, [k1dS, k2dS, g31dS, g32dS] = ESS_dS_parameters(EdS, f, k1seed, g31seed, Omega_r0h2, Omega_b0h2, Omega_c0h2, h)
        U0_list.append(U0)
        phi_prime0_list.append(0.9)
        Omega_m0_list.append(Omega_m0)
        Omega_r0_list.append(Omega_r0)
        Omega_l0_list.append(Omega_l0)
        k1_list.append(k1dS)
        k2_list.append(k2dS)
        g31_list.append(g31dS)
        g32_list.append(g32dS)
        
    U0_array = np.array( [1/i for i in EdS_array]  )
    np.savez(odeint_values_scan_filename, U0_array, [phiprime0], Omega_r0_list, Omega_m0_list, Omega_l0_list)
    
    np.savez(parameter_values_scan_filename, k1_list, k2_list, g31_list, g32_list)
    

odeint_values_scan_filename   = 'ESS-A_odeint_scanning_values'
parameter_values_scan_filename = 'ESS-A_parameter_scanning_values'
EdS_range = [0.8, 0.94, 2]
f_range = [0.0,1.0,1]
k1seed_range = [-5.7,-4.5,2]
g31seed_range = [-46.0,-46.0,1]
ESS_seed_to_direct_scanning_values(odeint_values_scan_filename, parameter_values_scan_filename, EdS_range, f_range, k1seed_range, g31seed_range)

##############################################################################################