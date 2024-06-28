# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import odeint
from scipy.optimize import fsolve
import sys
import os
import re
import matplotlib.pyplot as plt
import itertools as it
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from HiCOLA.Frontend.numerical_solver import solve_1stgrowth_factor, comp_Omega_m

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

def write_data_flex(data, output_filename_as_string, datanames=''):
    datafile_id = open(output_filename_as_string, 'wb')    #here you open the ascii file
    format_list = list(np.repeat('%.4e',len(data)))
    newdata = []
    for i in data:
        newdata.append(np.array(i[::]))
    realdata = np.array(newdata)
    realdata = realdata.T     #here you transpose your data to have it in columns
    np.savetxt(datafile_id, realdata, fmt=format_list, header=datanames)    #here the ascii file is populated.
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
    
def ESS_seed_to_column_scanning_values(scanning_parameters_filename, EdS_range, phi_range, phiprime_range, f_phi_range, k1seed_range, g31seed_range, Omega_r0h2_range, Omega_b0h2_range, Omega_c0h2_range, h = 0.7307, phiprime0 = 0.9):


    EdS_array = make_scan_array(*EdS_range)
    phi_array = make_scan_array(*phi_range)
    phiprime_array = make_scan_array(*phiprime_range)
    f_phi_array = make_scan_array(*f_phi_range)
    k1seed_array = make_scan_array(*k1seed_range)
    g31seed_array = make_scan_array(*g31seed_range)
    Omega_r0h2_array = make_scan_array(*Omega_r0h2_range)
    Omega_b0h2_array = make_scan_array(*Omega_b0h2_range)
    Omega_c0h2_array = make_scan_array(*Omega_c0h2_range)

    seed_cart_prod = it.product(EdS_array, phi_array, phiprime_array, f_phi_array, k1seed_array, g31seed_array, Omega_r0h2_array, Omega_b0h2_array, Omega_c0h2_array)
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
        EdS, phi0, phiprime0, f_phi, k1seed, g31seed, Omega_r0h2,Omega_b0h2, Omega_c0h2 = i
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

def littleH_to_massH0(h):
    #light speed in SI
    c = 2.998*1e+8

    #Newton's Gravitational constant in SI
    GN = 6.674*1e-11

    #Megaparsec in metres
    Mpc = 3.086*1e+22

    H0_mass_units = h*100.*1000.*c/Mpc/GN #used in k-mouflage screening factor
    return H0_mass_units

def EinsteinToJordan(a_E, H_E, betaK, phi_E, dphidx_E,M_pG4=1.0, M_sG4=1.0, H0_E = None, E_E_flag = False ):
    '''
    Converts Hubble rate in Einstein frame to Jordan frame. Assumes
    exponential conformal factor  A = exp(betaK*phi). Also gives numerical
    gradient of Jordan Hubble rate with respect to Jordan scale factor.
    '''
    
    A = np.array([np.exp(betaK*M_sG4*phi_Ev/M_pG4) for phi_Ev in phi_E])
    if E_E_flag is True:
        H_J =  H0_E*H_E*(1.0 + M_sG4*betaK*dphidx_E/M_pG4)/A
    else:
        #H_J = H_E*(1.0 + M_sG4*betaK*dphidx_E/M_pG4)/
        H_J = [H_Ev*(1.0 + M_sG4*betaK*dphidx_Ev/M_pG4)/Av for H_Ev, dphidx_Ev, Av in zip(H_E,dphidx_E,A)]
    a_J = A*a_E

    H0_J = H_J[-1]
    E_J = H_J/H0_J
    dEdaJ = InterpolatedUnivariateSpline(a_J,E_J, k=5).derivative()
    Eprime_J_wrtaJ_func = lambda a: dEdaJ(a)*a #convert dE/da to dE/d(lna)
    Eprime_J_wrtaJ = Eprime_J_wrtaJ_func(a_J)
    dEdaE = InterpolatedUnivariateSpline(a_E, E_J, k=5).derivative() #gradient w.r.t. a_E might be better for backend
    Eprime_J_wrtaE_func = lambda a: dEdaE(a)*a #convert dE/da to dE/d(lna)
    Eprime_J_wrtaE = Eprime_J_wrtaE_func(a_E)

    dHdaJ = InterpolatedUnivariateSpline(a_J,H_J, k=5).derivative()
    Hprime_J_wrtaJ_func = lambda a: dHdaJ(a)*a #convert dE/da to dE/d(lna)
    Hprime_J_wrtaJ = Hprime_J_wrtaJ_func(a_J)
    dHdaE = InterpolatedUnivariateSpline(a_E, H_J, k=5).derivative() #gradient w.r.t. a_E might be better for backend
    Hprime_J_wrtaE_func = lambda a: dHdaE(a)*a #convert dE/da to dE/d(lna)
    Hprime_J_wrtaE = Hprime_J_wrtaE_func(a_E)

    data = { "a_J":a_J, "E_J":E_J, "Eprime_J_wrt_a_J":Eprime_J_wrtaJ, "Eprime_J_wrt_a_E":Eprime_J_wrtaE, "H_J":H_J, "Hprime_J_wrt_a_J":Hprime_J_wrtaJ, "Hprime_J_wrt_a_E":Hprime_J_wrtaE,"H0_J":H0_J}

    return data

def compute_growth(bg_file, preforce_file, Omega_m0):
    a_arr, E_arr, EpE_arr = np.loadtxt(bg_file, unpack=True)
    aF_arr, chioverdelta_arr, coupling_arr = np.loadtxt(preforce_file, unpack=True)

    omegam_arr = comp_Omega_m(a_arr,Omega_m0,E_arr,timeswitch='scale_factor')

    D_arr, Dprime_arr = solve_1stgrowth_factor(a_arr,omegam_arr, EpE_arr,coupling_arr)

    return a_arr, D_arr, Dprime_arr

def growth_factor_breakdown(mg_bg_file, mg_preforce_file, Omega_m0, a_lcdm, E_lcdm):
    a_mg, E_mg, EpE_mg = np.loadtxt(mg_bg_file, unpack=True)
    a_mg, chioverdelta, coupling = np.loadtxt(mg_preforce_file,unpack=True)

    Elcdm_interp = InterpolatedUnivariateSpline(a_lcdm, E_lcdm, k=5)
    dEdalcdm_interp = Elcdm_interp.derivative()
    EprimeElcdm_func = lambda a: dEdalcdm_interp(a)*a/Elcdm_interp(a)

    #All modified effects
    omegam_arr = comp_Omega_m(a_mg, Omega_m0, E_mg, timeswitch='scale_factor')
    D_arr, Dprime_arr = solve_1stgrowth_factor(a_mg, omegam_arr, EpE_mg, coupling)

    #Isolate effect of E
    omegam_wLCDM_arr = comp_Omega_m(a_lcdm,Omega_m0,E_lcdm,timeswitch='scale_factor')
    omegam_wLCDM_interp = InterpolatedUnivariateSpline(a_lcdm, omegam_wLCDM_arr, k=5)
    D_noE_arr, Dprime_noE_arr = solve_1stgrowth_factor(a_mg,omegam_wLCDM_interp(a_mg), EpE_mg,coupling)

    Eeffect = D_arr/D_noE_arr

    #Isolate effect of Eprime
    D_noEprime_arr, Dprime_noEprime_arr = solve_1stgrowth_factor(a_mg, omegam_arr, EprimeElcdm_func(a_mg), coupling)
    Eprimeeffect = D_arr/D_noEprime_arr

    #Isolate effect of coupling
    D_nocoupling_arr, Dprime_nocoupling_arr = solve_1stgrowth_factor(a_mg, omegam_arr, EpE_mg,np.zeros(len(coupling)))
    couplingeffect = D_arr/D_nocoupling_arr

    #Isolate effect of E and Eprime (background)
    D_noEnoEpE_arr, Dprime_noEnoEpE_arr = solve_1stgrowth_factor(a_mg, omegam_wLCDM_interp(a_mg), EprimeElcdm_func(a_mg), coupling)
    backgroundeffect = D_arr/D_noEnoEpE_arr

    return a_mg, Eeffect, Eprimeeffect, couplingeffect, backgroundeffect

def import_pofk_by_z(root_directory,redshifts=["49.000","0.140","0.000"],dataheadings=['k','pofk','pofklin']):
    '''
    Imports FML power spectra files into a pandas dataframe.
    Expected directory structure:
    root_directory/
                   pofk*z<redshifts[0]>.txt
                   pofk*z<redshifts[1]>.txt
        .
        .
    Output: pandas dataframe, access through data[<float:redshift>][<string:"k"/"pofk"/"pofklin"]
    '''
    root_directory = Path(root_directory).resolve()
    files = os.listdir(root_directory)
    files = [file for file in files if "pofk" in file]

    dfs={}
    for filename in files:
        path_to_file = root_directory.joinpath(filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(path_to_file, header=0, delimiter='\s+',engine='python', names=dataheadings)
        df = df.set_index('k')
        match=re.search(r'_z(.*?)\.txt', str(filename))
        redshift=float(match.group(1))
        dfs[redshift]=df
    # Concatenate all DataFrames in the list into a single DataFrame
    data = pd.concat(dfs.values(),keys=dfs.keys(),axis=1)

    return data

def import_breakdown(root_directory, categories=["Full","QCDM","5thforcelinear","5thforceonly"],grlcdm=True,redshifts=["49.000","0.140","0.000"],dataheadings=['k','pofk','pofklin']):
    '''
    Imports FML power spectra files into a panda dataframe.
    Expected directory structure:
    root_directory/
                  /categories[0]/
                                /pofk*z<redshifts[0]>.txt
                                /pofk*z<redshifts[1]>.txt
                                .
                 /categories[1]
                               /pofk*z*<redshifts[0]>.txt etc.
    Output: pandas dataframe, access through data[<string: category>][<float: redshift>][<string: "k/pofk/pofklin">]                      
    '''
    root_directory = Path(root_directory).resolve()
    dfs={}
    for category in categories:
        subdirectory = root_directory.joinpath(category)
        df_category = import_pofk_by_z(subdirectory,redshifts=redshifts,dataheadings=dataheadings)
        dfs[category]=df_category
    if grlcdm is True:
        grlcdm_directory = root_directory.joinpath("../GRLCDM").resolve()
        df_grlcdm = import_pofk_by_z(grlcdm_directory,redshifts=redshifts,dataheadings=dataheadings)
        dfs["GRLCDM"]=df_grlcdm
    data = pd.concat(dfs.values(),keys=dfs.keys(),axis=1)
    return data

def redshift_check(z1, z2, nstep):
    '''
    Check that 2 redshifts are not closer than 1/nstep, the minimal allowed spacing for output redshifts for 
    the Hi-COLA backend.
    '''
    a = [1.0/(1.0+z) for z in [z1, z2]]
    boo = ( abs(a[0] - a[1]) < 1.0/nstep )
    return boo

def first_instance_cleaner(arr_dict,ref_key='a_E'):
    new_dict = {}
    for key in arr_dict.keys():
        new_dict[key] = np.array([])
    for vals in zip(*arr_dict.values()):
        if vals[0] not in new_dict[ref_key]:
            for val,key in zip(vals,arr_dict.keys()):
                new_dict[key] = np.append(new_dict[key], val)
    return new_dict