import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import numerical_solver as nta
import expression_builder as eb
import os
import time
import itertools as it
from multiprocessing import Pool

##############
symbol_decl = eb.declare_symbols()
exec(symbol_decl)
odeint_parameter_symbols = [E, phiprime, omegar, omegam]
######################


start = time.time()
N_proc = 4
##############
# dS testing #
##############

z_num_test = 1000 # number of redshifts to output at
z_ini_test = 9999. # early redshift to solve backwards to

####Number of points
Npoints = 3 #the old parameter that determines how many values of EdS and f_phi. I've kept it because I am still setting same no. for both, and naming files using it

Npoints_EdS = Npoints # how many EdS and f_phi values to test for
Npoints_f_phi = Npoints
Npoints_k1dS = 1
Npoints_g31dS = Npoints_k1dS

###Max and min values
EdS_max = 0.94
EdS_min = 0.5

f_phi_max = 1.0
f_phi_min = 0.0

k1_dS_max = 6.
k1_dS_min = -6.

g31_dS_max = 4.
g31_dS_min = -4.

### values based on fit to WMAP in table 1, 1306.3219
Omega_r0h2 = 4.28e-5
Omega_b0h2 = 0.02196
Omega_c0h2 = 0.1274
h_test = 0.7307

Omega_r0_test = Omega_r0h2/h_test/h_test
Omega_m0_test = (Omega_b0h2 + Omega_c0h2)/h_test/h_test
Omega_DE0_test = 1. - Omega_r0_test - Omega_m0_test
Omega_m_crit = 0.99 # matter must pass this value to be viable


GR_flag = False
suppression_flag = False
threshold = 0.
tolerance=1e-6
phi_prime0 = 0.9
cl_declaration = ['odeint_parameters',1]


early_DE_threshold=1.0
yellow_switch=False
blue_switch=False
red_switch=False

# define ranges of f_phi and E_dS to search over
E_dS_fac_arr = np.linspace(EdS_min, EdS_max, Npoints_EdS)
f_phi_arr = np.linspace(f_phi_min, f_phi_max, Npoints_f_phi)
k1_dS_arr = np.linspace(k1_dS_min,k1_dS_max,Npoints_k1dS)
g31_dS_arr = np.linspace(g31_dS_min,g31_dS_max,Npoints_g31dS)

############
# Specify reduced Horndeski theory #
############
k1, g31, g32, k2, = sym.symbols("k_1 g_{31} g_{32} k_2")
parameter_symbols = [k1,k2,g31,g32]


M_pG4_test, M_KG4_test, M_G3s_test, M_sG4_test, M_G3G4_test, M_Ks_test, M_gp_test = 1., 1., 1., 1., 1., 1., 1.
mass_ratio_list = [M_pG4_test, M_KG4_test, M_G3s_test, M_sG4_test, M_G3G4_test, M_Ks_test, M_gp_test]

K = k1*X + k2*X**2.
G3 = g31*X + g32*X**2.
G4 = 0.5
##################################################

description_string = 'Parallelised parameter scanner using pool.starmap on a functional form of scanning - PLL. No list on iter.prod. processes = '+str(N_proc)

##Initialise scan lists








# create lambdified functions for nta.run_solver_inv to use
model = 'ESS'
E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, fried_RHS_lambda, A_lambda, B2_lambda, \
    coupling_factor, alpha0_lambda, alpha1_lambda, alpha2_lambda, beta0_lambda, calB_lambda, calC_lambda = eb.create_Horndeski(K,G3,G4,parameter_symbols,mass_ratio_list)

lambda_functions = [E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, A_lambda, fried_RHS_lambda] #save into list for scanner


scan_list = it.product(E_dS_fac_arr,f_phi_arr,k1_dS_arr, g31_dS_arr,[cl_declaration],[early_DE_threshold],[yellow_switch],[blue_switch],[red_switch],
                       [tolerance],[z_num_test],[z_ini_test],[Omega_r0h2],[Omega_b0h2],[Omega_c0h2],
                       [h_test],[phi_prime0],[Omega_m_crit],[GR_flag],[suppression_flag],[threshold])
##---Setting up log files---
##To include date in log file names
time_snap = time.localtime()
folder_date = time.strftime("%Y-%m-%d",time_snap)
folder_time = time.strftime("%H-%M_%S", time_snap)
file_date = time.strftime("%Y-%m-%d_%H-%M-%S",time_snap)

##To create a directory to store logs if needed
if not os.path.exists("../scans"):
    os.makedirs("../scans")

##Sub-directory to organise files by date of creation
saving_directory ="../scans/"+folder_date+"/"

##Sub-directory to organise multiple runs performed on the same day by time of creation
saving_subdir = saving_directory+"/"+folder_time+"/"

if not os.path.exists(saving_directory):
    os.makedirs(saving_directory)
if not os.path.exists(saving_subdir):
    os.makedirs(saving_subdir)


path_to_txt_greens = saving_subdir+file_date+'_'+model+"_greens.txt"
path_to_txt_greys = saving_subdir+file_date+'_'+model+"_greys.txt"
path_to_txt_blacks = saving_subdir+file_date+'_'+model+"_blacks.txt"
path_to_txt_magentas = saving_subdir+file_date+'_'+model+"_magentas.txt"
path_to_txt_pinks = saving_subdir+file_date+'_'+model+"_pinks.txt"
path_to_txt_reds = saving_subdir+file_date+'_'+model+"_reds.txt"
path_to_txt_blues = saving_subdir+file_date+'_'+model+"_blues.txt"
path_to_txt_yellows = saving_subdir+file_date+'_'+model+"_yellows.txt"

# green_txt = open(path_to_txt_greens,"w")
# print('#'+str(['EdS','f_phi','k1dS-seed','g31dS-seed']),file=green_txt)

# black_txt = open(path_to_txt_blacks,"w")
# print('#'+str(['EdS','f_phi','k1dS-seed','g31dS-seed']),file=black_txt)

# magenta_txt = open(path_to_txt_magentas,"w")
# print('#'+str(['EdS','f_phi','k1dS-seed','g31dS-seed']),file=magenta_txt)

# red_txt = open(path_to_txt_reds,"w")
# print('#'+str(['EdS','f_phi','k1dS-seed','g31dS-seed']),file=red_txt)

# blue_txt = open(path_to_txt_blues,"w")
# print('#'+str(['EdS','f_phi','k1dS-seed','g31dS-seed']),file=blue_txt)

# yellow_txt = open(path_to_txt_yellows,"w")
# print('#'+str(['EdS','f_phi','k1dS-seed','g31dS-seed']),file=yellow_txt)

######################################

###Metadata file
path_to_metadata = saving_subdir+file_date+'_'+model+"_scan_metadata.txt"
scan_metadata_file = open(path_to_metadata,"w")

print(description_string,file=scan_metadata_file)
print('--- E_dS arguments ---',file=scan_metadata_file)
print('EdS_max='+str(EdS_max),file=scan_metadata_file)
print('EdS_min='+str(EdS_min),file=scan_metadata_file)
print('no. of EdS points='+str(Npoints_EdS),file=scan_metadata_file)
print('--- f_phi arguments ---',file=scan_metadata_file)
print('f_phi_max='+str(f_phi_max),file=scan_metadata_file)
print('f_phi_min='+str(f_phi_min),file=scan_metadata_file)
print('no. of f_phi points='+str(Npoints_f_phi),file=scan_metadata_file)
print('--- k1_dS arguments ---',file=scan_metadata_file)
print('k1dS_max='+str(k1_dS_max),file=scan_metadata_file)
print('k1dS_min='+str(k1_dS_min),file=scan_metadata_file)
print('no. of k1dS points='+str(Npoints_k1dS),file=scan_metadata_file)
print('--- g31_dS arguments ---',file=scan_metadata_file)
print('g31dS_max='+str(g31_dS_max),file=scan_metadata_file)
print('g31dS_min='+str(g31_dS_min),file=scan_metadata_file)
print('no. of g31dS points='+str(Npoints_g31dS),file=scan_metadata_file)
print('--- Scan parameters ---',file=scan_metadata_file)
print('No. of processors='+str(N_proc),file=scan_metadata_file)
print('threshold (for replacing infinitesimal A)='+str(threshold),file=scan_metadata_file)
print('z_num='+str(z_num_test),file=scan_metadata_file)
print('z_max='+str(z_ini_test),file=scan_metadata_file)
print('early_DE_threshold (determines where pink check starts)='+str(early_DE_threshold),file=scan_metadata_file)
print('phi_prime0='+str(phi_prime0),file=scan_metadata_file)
print('tolerance (for black criterion)='+str(tolerance),file=scan_metadata_file)
print('--- Cosmological parameters ---',file=scan_metadata_file)
print('Omega_m0='+str(Omega_m0_test),file=scan_metadata_file)
print('Omega_r0='+str(Omega_r0_test),file=scan_metadata_file)
print('Omega_m_crit='+str(Omega_m_crit),file=scan_metadata_file)
print('--- Criteria switches ---',file=scan_metadata_file)
print('yellow_switch='+str(yellow_switch),file=scan_metadata_file)
print('blue_switch='+str(blue_switch),file=scan_metadata_file)
print('red_switch='+str(red_switch),file=scan_metadata_file)


# initialise the figure
# figB, axB = plt.subplots()
# axB.set_title('Bill')
# figA, axA = plt.subplots()
# axA.set_title('Ashim')



def parameter_scanner(EdS, f_phi, k1seed, g31seed,cl_declaration = ['odeint_parameters',1],
                      early_DE_threshold=1.0, yellow_switch=False, blue_switch=False, red_switch=False,tolerance=10e-6,
                      znum=1000,zmax=9999.,Omega_r0h2 = 4.28e-5,Omega_b0h2 = 0.02196,Omega_c0h2 = 0.1274,h_test = 0.7307,phi_prime0=0.9,Omega_m_crit=0.99,
                      GR_flag = False, suppression_flag = False, c_M_test = 0., hmaxvv = None, threshold = 0.):
    Omega_r0 = Omega_r0h2/h_test/h_test
    Omega_m0 = (Omega_b0h2 + Omega_c0h2)/h_test/h_test
    Omega_DE0 = 1. - Omega_m0 - Omega_r0
    Omega_l0 = (1.-f_phi)*Omega_DE0

    # U0 = 1./EdS
    # #[E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, A_lambda, fried_RHS_lambda] = [*lambda_functions]
    # alpha_expr = 1.-Omega_l0/EdS/EdS
    # k1_dS = k1seed*alpha_expr         #k1_dSv has been called k1(dS)-seed in my notes
    # k2_dS = -2.*k1_dS - 12.*(1. - Omega_l0/EdS/EdS)
    # g31_dS = g31seed*alpha_expr             #g31_dSv is g31(dS)-seed
    # g32_dS = 0.5*( 1.- ( Omega_l0/EdS/EdS + k1_dS/6. + k2_dS/4. + g31_dS) )
    # parameters = [k1_dS,k2_dS, g31_dS, g32_dS]
    # print(parameters)
    # a_arr_invA, U_arrA, U_prime_U_arrA, U_prime_arrA , y_arrA, y_prime_arrA, Omega_r_arrA, Omega_m_arrA, Omega_DE_arrA, Omega_L_arrA, Omega_phi_arrA, Omega_phi_diff_arrA, Omega_r_prime_arrA,Omega_m_prime_arrA, Omega_L_prime_arrA,  A_arrA= nta.run_solver_inv(z_num_test, z_ini_test, U0, phi_prime0, E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, A_lambda, fried_RHS_lambda, cl_declaration, parameters, Omega_r0, Omega_m0, Omega_l0, c_M_test, hmaxvv, suppression_flag, threshold,GR_flag)

    # alpha_facA = 1.-Omega_L_arrA[0]/EdS/EdS

    # trackA = B2_lambda(U_arrA[0],y_arrA[0],*parameters) #general version
    # maxdensity_list = []
    # mindensity_list = []
    # for density_arr in [Omega_m_arrA, Omega_r_arrA, Omega_L_arrA, Omega_phi_arrA, Omega_DE_arrA]:
    #     maxdensity_list.append(np.max(density_arr))
    #     mindensity_list.append(np.min(density_arr))
    # Omega_m_max, Omega_r_max, Omega_L_max, Omega_phi_max, Omega_DE_max = maxdensity_list
    # Omega_m_min, Omega_r_min, Omega_L_min, Omega_phi_min, Omega_DE_min = mindensity_list
    fv = f_phi
    Ev = EdS
    U0v = 1./Ev
    k1_dSv = k1seed
    g31_dSv = g31seed

    Omega_l0_test = Omega_l0


    Omega_l0_test = (1.-fv)*Omega_DE0_test
    alpha_expr = 1.-Omega_l0_test/Ev/Ev

    k1_dS = k1_dSv*alpha_expr         #k1_dSv has been called k1(dS)-seed in my notes
    k2_dS = -2.*k1_dS - 12.*(1. - Omega_l0_test/Ev/Ev)
    g31_dS = g31_dSv*alpha_expr#2.*alpha_expr                 #g31_dSv is g31(dS)-seed
    g32_dS = 0.5*( 1.- ( Omega_l0_test/Ev/Ev + k1_dS/6. + k2_dS/4. + g31_dS) )
    parameters = [k1_dS,k2_dS, g31_dS, g32_dS]

    a_arr_invA, U_arrA, U_prime_U_arrA, U_prime_arrA , y_arrA, y_prime_arrA, Omega_r_arrA, Omega_m_arrA, Omega_DE_arrA, Omega_L_arrA, Omega_phi_arrA, Omega_phi_diff_arrA, Omega_r_prime_arrA,Omega_m_prime_arrA, Omega_L_prime_arrA,  A_arrA, calB_arr, calC_arr, coupling_factor_arr,chioverdelta_arr = nta.run_solver_inv(z_num_test, z_ini_test, U0v, phi_prime0, E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, A_lambda, fried_RHS_lambda, calB_lambda, calC_lambda, coupling_factor, cl_declaration, parameters, Omega_r0_test, Omega_m0_test, Omega_l0_test, parameter_symbols, odeint_parameter_symbols, suppression_flag, threshold,GR_flag)

    alpha_facA = 1.-Omega_L_arrA[0]/Ev/Ev

    trackA = B2_lambda(U_arrA[0],y_arrA[0],*parameters) #general version
    maxdensity_list = []
    mindensity_list = []
    for density_arr in [Omega_m_arrA, Omega_r_arrA, Omega_L_arrA, Omega_phi_arrA, Omega_DE_arrA]:
        maxdensity_list.append(np.max(density_arr))
        mindensity_list.append(np.min(density_arr))
    Omega_m_max, Omega_r_max, Omega_L_max, Omega_phi_max, Omega_DE_max = maxdensity_list
    Omega_m_min, Omega_r_min, Omega_L_min, Omega_phi_min, Omega_DE_min = mindensity_list

    z_arr_inv = [(1-a)/a for a in a_arr_invA]
    early_DE_index = nta.nearest_index(z_arr_inv,early_DE_threshold)
    early_Omega_DE_arr = Omega_DE_arrA[early_DE_index:]
    early_Omega_DE_max = np.max(early_Omega_DE_arr)

    if alpha_facA < 0 and yellow_switch==True:
        with open(path_to_txt_yellows,"a") as yellow_txt:
            yellow_txt.write(str([EdS,f_phi,k1seed,g31seed])+"\n")
    elif trackA < 0 and blue_switch==True:
        with open(path_to_txt_blues,"a") as blue_txt:
            blue_txt.write(str([EdS,f_phi,k1seed,g31seed])+"\n")
    elif early_Omega_DE_max > Omega_DE_arrA[0]: #less harsh early DE criterion, only checks up until redshift = early_DE_threshold
        with open(path_to_txt_pinks,"a") as pink_txt:
            pink_txt.write(str([EdS,f_phi,k1seed,g31seed, early_Omega_DE_max])+"\n")
    elif Omega_DE_max > Omega_DE_arrA[0] and red_switch==True:
        with open(path_to_txt_reds,"a") as red_txt:
            red_txt.write(str([EdS,f_phi,k1seed,g31seed,Omega_DE_max])+"\n")
    elif Omega_m_max < Omega_m_crit:
        with open(path_to_txt_magentas,"a") as magenta_txt:
            magenta_txt.write(str([EdS,f_phi,k1seed,g31seed])+"\n")
    elif ( (Omega_m_max > 1.0 + tolerance) or (Omega_m_min < 0.-tolerance) or
         (Omega_r_max > 1.0+ tolerance) or (Omega_r_min < 0.-tolerance) or
         (Omega_L_max > 1.0+ tolerance) or (Omega_L_min < 0.-tolerance) or
         (Omega_phi_max > 1.0+ tolerance) or (Omega_phi_min < 0.-tolerance) or
         (Omega_DE_max > 1.0+ tolerance) or (Omega_DE_min < 0.-tolerance)  ):
        with open(path_to_txt_blacks,"a") as black_txt:
            black_txt.write(str([EdS,f_phi,k1seed,g31seed])+"\n")
    elif ( (np.isnan(Omega_m_max)) or
         (np.isnan(Omega_r_max)) or
         (np.isnan(Omega_L_max)) or
         (np.isnan(Omega_phi_max)) or
         (np.isnan(Omega_DE_max))            ):
        with open(path_to_txt_greys,"a") as grey_txt:
            grey_txt.write(str([EdS,f_phi,k1seed,g31seed])+"\n")
    else:
        with open(path_to_txt_greens,"a") as green_txt:
            green_txt.write(str([EdS,f_phi,k1seed,g31seed])+"\n")

# loop over search variables

# for i in list(scan_list):
#     EdSv = i[0]
#     fv = i[1]
#     k1seedv = i[2]
#     g31seedv = i[3]
#     parameter_scanner(EdSv, fv, k1seedv, g31seedv)


if __name__ == '__main__':
    pool = Pool(processes=N_proc)
    inputs = scan_list
    pool.starmap(parameter_scanner, inputs)

## save and close scan files
# scan_lists = [green_arr,black_arr,magenta_arr,red_arr,blue_arr,yellow_arr]
# files = [green_txt,black_txt,magenta_txt,red_txt,blue_txt,yellow_txt]

## I am not saving the scan_lists directly for formatting reasons, since each scan_list is a list, containing lists of length 4. I
## don't get linebreaks following each entry in scan_list, making txtfile hard to read.
# for scan_list,txtfile in zip(scan_lists,files):
#     for i in scan_list:
#         if i is not None:
#             print(i,file=txtfile)
#     txtfile.close()
end = time.time()
print('--- Runtime information ---',file=scan_metadata_file)
print("Script duration: "+str(end-start),file=scan_metadata_file)

scan_metadata_file.close()
