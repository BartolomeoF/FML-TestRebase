# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import sys
import matplotlib.pyplot as plt
import itertools as it
from scipy.interpolate import CubicSpline

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

def ESS_dS_parameters(EdS, f_phi, k1seed, g31seed,Omega_r0h2 = 4.28e-5, Omega_b0h2 = 0.02196, Omega_c0h2 = 0.1274, h = 0.7307):
    Omega_r0 = Omega_r0h2/h/h
    Omega_m0 = (Omega_b0h2 + Omega_c0h2)/h/h
    Omega_DE0 = 1. - Omega_r0 - Omega_m0
    Omega_l0 = (1.-f_phi)*Omega_DE0

    U0 = 1./EdS

    alpha_expr = 1.-Omega_l0/EdS/EdS

    k1_dS = k1seed*alpha_expr         #k1_dSv has been called k1(dS)-seed in my notes
    k2_dS = -2.*k1_dS - 12.*alpha_expr
    g31_dS = g31seed*alpha_expr#2.*alpha_expr                 #g31_dSv is g31(dS)-seed
    g32_dS = 0.5*( 1.- ( Omega_l0/EdS/EdS + k1_dS/6. + k2_dS/4. + g31_dS) )
    parameters = [k1_dS,k2_dS, g31_dS, g32_dS]
    return U0, Omega_r0, Omega_m0, Omega_l0, parameters

def ESS_seed_to_direct_scanning_values(scanning_parameters_filename, EdS_range, phi_range, phiprime_range, f_phi_range, k1seed_range, g31seed_range, Omega_r0h2 = 4.28e-5, Omega_b0h2 = 0.02196, Omega_c0h2 = 0.1274, h = 0.7307, phiprime0 = 0.9):


    EdS_array = make_scan_array(*EdS_range)
    phi_array = make_scan_array(*phi_range)
    phiprime_array = make_scan_array(*phiprime_range)
    f_phi_array = make_scan_array(*f_phi_range)
    k1seed_array = make_scan_array(*k1seed_range)
    g31seed_array = make_scan_array(*g31seed_range)

    seed_cart_prod = it.product(EdS_array, phi_array, phiprime_array, f_phi_array, k1seed_array, g31seed_array)
    seed_cart_prod2 = it.product(EdS_array, phi_array, phiprime_array, f_phi_array, k1seed_array, g31seed_array)
    # print(len(list(seed_cart_prod2)))
    scan_list = []
    # U0_list = []
    # phi_prime0_list = []
    # Omega_m0_list = []
    # Omega_r0_list = []
    # Omega_l0_list = []
    # k1_list = []
    # k2_list = []
    # g31_list = []
    # g32_list = []
    for i in seed_cart_prod:
        EdS, phi0, phiprime0, f_phi, k1seed, g31seed = i
        U0, Omega_r0, Omega_m0, Omega_l0, [k1dS, k2dS, g31dS, g32dS] = ESS_dS_parameters(EdS, f_phi, k1seed, g31seed, Omega_r0h2, Omega_b0h2, Omega_c0h2, h)
        scan_list_entry = [U0, phi0, phiprime0, Omega_r0, Omega_m0, Omega_l0, k1dS, k2dS, g31dS, g32dS ] #scan_list_entry[5:] = parameters
        scan_list.append(scan_list_entry)
        scan_array = np.array(scan_list)
    # print(scan_list) 
    # print(len(scan_list))
    np.save(scanning_parameters_filename, scan_array)
    
def ESS_seed_to_column_scanning_values(scanning_parameters_filename, EdS_range, phi_range, phiprime_range, f_phi_range, k1seed_range, g31seed_range, Omega_r0h2 = 4.28e-5, Omega_b0h2 = 0.02196, Omega_c0h2 = 0.1274, h = 0.7307, phiprime0 = 0.9):


    EdS_array = make_scan_array(*EdS_range)
    phi_array = make_scan_array(*phi_range)
    phiprime_array = make_scan_array(*phiprime_range)
    f_phi_array = make_scan_array(*f_phi_range)
    k1seed_array = make_scan_array(*k1seed_range)
    g31seed_array = make_scan_array(*g31seed_range)

    seed_cart_prod = it.product(EdS_array, phi_array, phiprime_array, f_phi_array, k1seed_array, g31seed_array)
    seed_cart_prod2 = it.product(EdS_array, phi_array, phiprime_array, f_phi_array, k1seed_array, g31seed_array)
    # print(len(list(seed_cart_prod2)))
    scan_list = []
    # U0_list = []
    # phi_prime0_list = []
    # Omega_m0_list = []
    # Omega_r0_list = []
    # Omega_l0_list = []
    # k1_list = []
    # k2_list = []
    # g31_list = []
    # g32_list = []
    U0_column = []
    phi0_column = []
    phiprime0_column = []
    k1_column = []
    k2_column = []
    g31_column = []
    g32_column = []
    Omega_r0_column = []
    Omega_m0_column = []
    Omega_l0_column = []
    for i in seed_cart_prod:
        EdS, phi0, phiprime0, f_phi, k1seed, g31seed = i
        U0, Omega_r0, Omega_m0, Omega_l0, [k1dS, k2dS, g31dS, g32dS] = ESS_dS_parameters(EdS, f_phi, k1seed, g31seed, Omega_r0h2, Omega_b0h2, Omega_c0h2, h)
        # scan_list_entry = [U0, phiprime0, Omega_r0, Omega_m0, Omega_l0, k1dS, k2dS, g31dS, g32dS ] #scan_list_entry[5:] = parameters
        # scan_list.append(scan_list_entry)
        # scan_array = np.array(scan_list)
        U0_column.append(U0)
        phi0_column.append(phi0)
        phiprime0_column.append(phiprime0)
        k1_column.append(k1dS)
        k2_column.append(k2dS)
        g31_column.append(g31dS)
        g32_column.append(g32dS)
        Omega_r0_column.append(Omega_r0)
        Omega_m0_column.append(Omega_m0)
        Omega_l0_column.append(Omega_l0)
    # print(scan_list) 
    # print(len(scan_list))
    np.savetxt(scanning_parameters_filename, np.transpose([U0_column, phi0_column, phiprime0_column, Omega_r0_column, Omega_m0_column, Omega_l0_column, k1_column, k2_column, g31_column, g32_column]))

def ESS_direct_to_seed(k1, g31, omega_l0, f_phi, EdS):
    alpha = 1. - omega_l0/EdS/EdS
    k1seed = k1/alpha
    g31seed = g31/alpha

    return k1seed, g31seed, omega_l0, f_phi, EdS


def ESS_seed_to_direct(k1seed, g31seed, omega_l0,f_phi,EdS):
    alpha = 1. - omega_l0/EdS/EdS
    k1 = k1seed*alpha
    g31 = g31seed*alpha
    return k1, g31

def renamer(filename):
    if filename[-6] == "_":
        counter = eval(filename[-5]) + 1 #if you get NameError: name '_' is not defined, 
                                         #easiest fix is to change name of expansion/force file
                                         #this function expects names that look like 'abc.txt' or 'abc_1.txt'
                                         #not 'abc_d.txt', where d is a non-integer character.
        filename = filename[:-5] + f"{counter}.txt"
    else:
        filename = filename[:-4] + "_1.txt"
    return filename


def compute_fphi(omega_l, omega_m, omega_r):
    omega_DE = 1. - omega_m - omega_r
    f_phi = 1. - omega_l/omega_DE
    return f_phi

def interpolate_common_domain(k1_arr, pofk1_arr, k2_arr, pofk2_arr):
    '''
    Finds common domain between k1_arr and k2_arr and splines pofk2_arr over
    the trimmed k1_arr
    '''
    
    k1max = np.max(k1_arr)
    k1min = np.min(k1_arr)
    
    k2max = np.max(k2_arr)
    k2min = np.min(k2_arr)
    
    if k1max < k2max:
        upperbound = k1max
        k2_arr = k2_arr[k2_arr < upperbound]
    elif k2max <= k1max:
        upperbound = k2max
        k1_arr = k1_arr[k1_arr < upperbound]
    if k1min < k2min:
        lowerbound = k2min
        k1_arr = k1_arr[lowerbound < k1_arr]
    elif k2min <= k1min:
        lowerbound = k1min
        k2_arr = k2_arr[lowerbound < k2_arr]
        
    pofk2_spline = CubicSpline(k2_arr, pofk2_arr)
    pofk2_splined_arr = pofk2_spline(k1_arr)
    
    return k1_arr, pofk1_arr, pofk2_splined_arr