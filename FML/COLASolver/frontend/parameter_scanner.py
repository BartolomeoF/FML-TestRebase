import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import numerical_solver as ns
import expression_builder as eb
import os
import time
import itertools as it
from multiprocessing import Pool
from read_parameters import read_in_scan_parameters, read_in_scan_settings
from argparse import ArgumentParser
from configobj import ConfigObj


##############
symbol_decl = eb.declare_symbols()
exec(symbol_decl)
odeint_parameter_symbols = [E, phiprime, omegar, omegam]
######################


start = time.time()
N_proc = 1
##############
# dS testing #
##############

# z_num_test = 1000 # number of redshifts to output at
# z_ini_test = 9999. # early redshift to solve backwards to

# ####Number of points
# Npoints = 3 #the old parameter that determines how many values of EdS and f_phi. I've kept it because I am still setting same no. for both, and naming files using it

# Npoints_EdS = Npoints # how many EdS and f_phi values to test for
# Npoints_f_phi = Npoints
# Npoints_k1dS = 1
# Npoints_g31dS = Npoints_k1dS

# ###Max and min values
# EdS_max = 0.94
# EdS_min = 0.5

# f_phi_max = 1.0
# f_phi_min = 0.0

# k1_dS_max = 6.
# k1_dS_min = -6.

# g31_dS_max = 4.
# g31_dS_min = -4.

# ### values based on fit to WMAP in table 1, 1306.3219
# Omega_r0h2 = 4.28e-5
# Omega_b0h2 = 0.02196
# Omega_c0h2 = 0.1274
# h_test = 0.7307

# Omega_r0_test = Omega_r0h2/h_test/h_test
# Omega_m0_test = (Omega_b0h2 + Omega_c0h2)/h_test/h_test
# Omega_DE0_test = 1. - Omega_r0_test - Omega_m0_test
# Omega_m_crit = 0.99 # matter must pass this value to be viable


# GR_flag = False
# suppression_flag = False
# threshold = 0.
# tolerance=1e-6
# phi_prime0 = 0.9
# cl_declaration = ['odeint_parameters',1]


# early_DE_threshold=1.0
# yellow_switch=False
# blue_switch=False
# red_switch=False

#####################

parser = ArgumentParser(prog='Scanner')

parser.add_argument('input_ini_filenames',nargs=2)
parser.add_argument('-d', '--direct',action='store_true')

args = parser.parse_args()
print(args)
filenames = args.input_ini_filenames
Horndeski_scanning_path = filenames[0]
numerical_scanning_path = filenames[1]
direct_scan_read_flag = args.direct
print(direct_scan_read_flag)

if direct_scan_read_flag is True:
    odeint_scan_dict = dict( np.load(numerical_scanning_path) )
    # odeint_scan_lists = [i for i in odeint_scan_dict.values()]
    # odeint_scan_arrays = np.array(odeint_scan_lists)
    # odeint_scan_arraysT = odeint_scan_arrays.T

    parameter_scan_dict = dict( np.load(Horndeski_scanning_path) )
    # parameter_scan_lists = [i for i in parameter_scan_dict.values()]
    # parameter_scan_arrays = np.array(parameter_scan_lists)
    # parameter_scan_arraysT = parameter_scan_arrays.T


    U0_array = odeint_scan_dict['arr_0']
    phiprime0_array = odeint_scan_dict['arr_1']
    Omega_r0_array = odeint_scan_dict['arr_2']
    Omega_m0_array = odeint_scan_dict['arr_3']
    Omega_l0_array = odeint_scan_dict['arr_4']

    parameter_arrays = []
    for i in parameter_scan_dict.values():
        parameter_arrays.append(i)
    parameter_arrays = np.array(parameter_arrays)
    parameter_arrays = parameter_arrays.T

    read_out_dict = read_in_scan_settings('scan_settings.ini')
    odeint_parameter_symbols = [E, phiprime, omegar, omegam]
    read_out_dict.update({'odeint_parameter_symbols':odeint_parameter_symbols})
    print('testtsetsetse')
    print(read_out_dict)
    print('individual arrays')
    print(U0_array)
    print(phiprime0_array)
    print(Omega_r0_array)
    print(Omega_m0_array)
    print(Omega_l0_array)
    print(parameter_arrays)
    
    #parameter_cartesian_product = it.product(*parameter_arrays)
    #print(U0_array, phiprime0_array, Omega_r0_array, Omega_m0_array, Omega_l0_array, parameter_arrays, [read_out_dict])
    scan_list = it.product(U0_array, phiprime0_array, Omega_r0_array, Omega_m0_array, Omega_l0_array, parameter_arrays, [read_out_dict])

else:
    read_out_dict = read_in_scan_parameters(Horndeski_scanning_path, numerical_scanning_path)
    odeint_parameter_symbols = [E, phiprime, omegar, omegam]
    read_out_dict.update({'odeint_parameter_symbols':odeint_parameter_symbols})


    K = read_out_dict['K']
    G3 = read_out_dict['G3']
    G4 = read_out_dict['G4']
    cosmology_name = read_out_dict['cosmo_name']

    # [Npoints, z_max, supp_flag, threshold_value, GR_flag] = read_out_dict['simulation_parameters']
    # red_switch = read_out_dict['red_switch']
    # blue_switch = read_out_dict['blue_switch']
    # yellow_switch = read_out_dict['yellow_switch']
    # tolerance = read_out_dict['tolerance']
    # early_DE_threshold = read_out_dict['early_DE_threshold']
    mass_ratio_list = read_out_dict['mass_ratio_list']
    symbol_list = read_out_dict['symbol_list']
    # closure_declaration = read_out_dict['closure_declaration']

    [Omega_r0_array, Omega_m0_array, Omega_l0_array] = read_out_dict.pop('cosmological_parameter_arrays')
    [U0_array, phi_prime0_array] = read_out_dict.pop('initial_condition_arrays')
    parameter_arrays = read_out_dict.pop('Horndeski_parameter_arrays')

    # define ranges of f_phi and E_dS to search over
    # E_dS_fac_arr = np.linspace(EdS_min, EdS_max, Npoints_EdS)
    # f_phi_arr = np.linspace(f_phi_min, f_phi_max, Npoints_f_phi)
    # k1_dS_arr = np.linspace(k1_dS_min,k1_dS_max,Npoints_k1dS)
    # g31_dS_arr = np.linspace(g31_dS_min,g31_dS_max,Npoints_g31dS)



    ##Initialise scan lists

    print(U0_array)
    print(phi_prime0_array)
    print(Omega_r0_array)
    print(Omega_m0_array)
    print(Omega_l0_array)
    print(parameter_arrays)





    parameter_cartesian_product = it.product(*parameter_arrays)


    scan_list = it.product(U0_array, phi_prime0_array, Omega_r0_array,Omega_m0_array,Omega_l0_array,list(parameter_cartesian_product), [read_out_dict])
    parameter_cartesian_product = it.product(*parameter_arrays)
    scan_list2 = it.product(U0_array, phi_prime0_array, Omega_r0_array,Omega_m0_array,Omega_l0_array,list(parameter_cartesian_product), [read_out_dict])
    print('scan_list2')
    scan_list_to_print = list(scan_list2)
    print(len(scan_list_to_print))
# for i in scan_list_to_print:
#     print(i)
##---Setting up log files---




# scan_metadata_file = open(path_to_metadata,"w")

# print(description_string,file=scan_metadata_file)
# print('--- E_dS arguments ---',file=scan_metadata_file)
# print('EdS_max='+str(EdS_max),file=scan_metadata_file)
# print('EdS_min='+str(EdS_min),file=scan_metadata_file)
# print('no. of EdS points='+str(Npoints_EdS),file=scan_metadata_file)
# print('--- f_phi arguments ---',file=scan_metadata_file)
# print('f_phi_max='+str(f_phi_max),file=scan_metadata_file)
# print('f_phi_min='+str(f_phi_min),file=scan_metadata_file)
# print('no. of f_phi points='+str(Npoints_f_phi),file=scan_metadata_file)
# print('--- k1_dS arguments ---',file=scan_metadata_file)
# print('k1dS_max='+str(k1_dS_max),file=scan_metadata_file)
# print('k1dS_min='+str(k1_dS_min),file=scan_metadata_file)
# print('no. of k1dS points='+str(Npoints_k1dS),file=scan_metadata_file)
# print('--- g31_dS arguments ---',file=scan_metadata_file)
# print('g31dS_max='+str(g31_dS_max),file=scan_metadata_file)
# print('g31dS_min='+str(g31_dS_min),file=scan_metadata_file)
# print('no. of g31dS points='+str(Npoints_g31dS),file=scan_metadata_file)
# print('--- Scan parameters ---',file=scan_metadata_file)
# print('No. of processors='+str(N_proc),file=scan_metadata_file)
# print('threshold (for replacing infinitesimal A)='+str(threshold),file=scan_metadata_file)
# print('z_num='+str(z_num_test),file=scan_metadata_file)
# print('z_max='+str(z_ini_test),file=scan_metadata_file)
# print('early_DE_threshold (determines where pink check starts)='+str(early_DE_threshold),file=scan_metadata_file)
# print('phi_prime0='+str(phi_prime0),file=scan_metadata_file)
# print('tolerance (for black criterion)='+str(tolerance),file=scan_metadata_file)
# print('--- Cosmological parameters ---',file=scan_metadata_file)
# print('Omega_m0='+str(Omega_m0_test),file=scan_metadata_file)
# print('Omega_r0='+str(Omega_r0_test),file=scan_metadata_file)
# print('Omega_m_crit='+str(Omega_m_crit),file=scan_metadata_file)
# print('--- Criteria switches ---',file=scan_metadata_file)
# print('yellow_switch='+str(yellow_switch),file=scan_metadata_file)
# print('blue_switch='+str(blue_switch),file=scan_metadata_file)
# print('red_switch='+str(red_switch),file=scan_metadata_file)


# initialise the figure
# figB, axB = plt.subplots()
# axB.set_title('Bill')
# figA, axA = plt.subplots()
# axA.set_title('Ashim')

model = read_out_dict['model_name']

##################################################

description_string = 'Parallelised parameter scanner using pool.starmap on a functional form of scanning - PLL. No list on iter.prod. processes = '+str(N_proc)


##To include date in log file names
time_snap = time.localtime()
folder_date = time.strftime("%Y-%m-%d",time_snap)
folder_time = time.strftime("%H-%M_%S", time_snap)
file_date = time.strftime("%Y-%m-%d_%H-%M-%S",time_snap)

##To create a directory to store logs if needed
if not os.path.exists("./scans"):
    os.makedirs("./scans")

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
read_out_dict_copy = read_out_dict.copy()
print(read_out_dict_copy)
write_dict = ConfigObj(read_out_dict_copy)
print(write_dict)
write_dict.filename = path_to_metadata
write_dict.write()

def parameter_scanner(U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, parameters, read_out_dict):

# (EdS, f_phi, k1seed, g31seed,cl_declaration = ['odeint_parameters',1],
#                       early_DE_threshold=1.0, yellow_switch=False, blue_switch=False, red_switch=False,tolerance=10e-6,
#                       znum=1000,zmax=9999.,Omega_r0h2 = 4.28e-5,Omega_b0h2 = 0.02196,Omega_c0h2 = 0.1274,h_test = 0.7307,phi_prime0=0.9,Omega_m_crit=0.99,
#                       GR_flag = False, suppression_flag = False, c_M_test = 0., hmaxvv = None, threshold = 0.):
#     Omega_r0 = Omega_r0h2/h_test/h_test


    # [Omega_r0, Omega_m0, Omega_l0] = read_out_dict['cosmological_parameters']
    # [E0, phi_prime0] = read_out_dict['initial_conditions']
    # [Npoints, z_max, suppression_flag, threshold, GR_flag] = read_out_dict['simulation_parameters']
    # parameters = read_out_dict['Horndeski_parameters']

    # E_prime_E_lambda = read_out_dict['E_prime_E_lambda']
    # E_prime_E_safelambda = read_out_dict['E_prime_E_safelambda']
    # phi_primeprime_lambda = read_out_dict['phi_primeprime_lambda']
    # phi_primeprime_safelambda = read_out_dict['phi_primeprime_safelambda']
    # omega_phi_lambda = read_out_dict['omega_phi_lambda']
    # A_lambda = read_out_dict['A_lambda']
    # fried_RHS_lambda = read_out_dict['fried_RHS_lambda']
    # calB_lambda = read_out_dict['calB_lambda']
    # calC_lambda = read_out_dict['calC_lambda']
    # coupling_factor = read_out_dict['coupling_factor']

    # parameter_symbols = read_out_dict['symbol_list']
    # odeint_parameter_symbols = read_out_dict['odeint_parameter_symbols']
    # cl_declaration = read_out_dict['closure_declaration']



    red_switch = read_out_dict['red_switch']
    blue_switch = read_out_dict['blue_switch']
    yellow_switch = read_out_dict['yellow_switch']
    tolerance = read_out_dict['tolerance']
    early_DE_threshold = read_out_dict['early_DE_threshold']
    Omega_m_crit = read_out_dict['Omega_m_crit']

    K = read_out_dict['K']
    G3 = read_out_dict['G3']
    G4 = read_out_dict['G4']
    mass_ratio_list = read_out_dict['mass_ratio_list']
    symbol_list = read_out_dict['symbol_list']

    lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list)
    E_prime_E_lambda = lambdified_functions['E_prime_E_lambda']
    B2_lambda = lambdified_functions['B2_lambda']
    E_prime_E_lambda(1,1,1,1,1,1,1,1)
    read_out_dict.update(lambdified_functions)

    print(parameters)

    cosmological_parameters = [Omega_r0, Omega_m0, Omega_l0]
    initial_conditions = [U0, phi_prime0]
    repack_dict = {'cosmological_parameters':cosmological_parameters, 'initial_conditions':initial_conditions, 'Horndeski_parameters':parameters}
    read_out_dict.update(repack_dict)



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


    background_quantities = ns.run_solver(read_out_dict)

    Omega_L_arrA = background_quantities['omega_l']
    Omega_r_arrA = background_quantities['omega_r']
    Omega_m_arrA = background_quantities['omega_m']
    Omega_phi_arrA = background_quantities['omega_phi']
    Omega_DE_arrA = background_quantities['omega_DE']
    a_arr_invA = background_quantities['a']
    U_arrA = background_quantities['Hubble']
    y_arrA = background_quantities['scalar_prime']

    alpha_facA = 1.-Omega_L_arrA[0]*U0*U0

    trackA = B2_lambda(U_arrA[0],y_arrA[0],*parameters) #general version
    maxdensity_list = []
    mindensity_list = []
    for density_arr in [Omega_m_arrA, Omega_r_arrA, Omega_L_arrA, Omega_phi_arrA, Omega_DE_arrA]:
        maxdensity_list.append(np.max(density_arr))
        mindensity_list.append(np.min(density_arr))
    Omega_m_max, Omega_r_max, Omega_L_max, Omega_phi_max, Omega_DE_max = maxdensity_list
    Omega_m_min, Omega_r_min, Omega_L_min, Omega_phi_min, Omega_DE_min = mindensity_list

    z_arr_inv = [(1-a)/a for a in a_arr_invA]
    early_DE_index = ns.nearest_index(z_arr_inv,early_DE_threshold)
    early_Omega_DE_arr = Omega_DE_arrA[early_DE_index:]
    early_Omega_DE_max = np.max(early_Omega_DE_arr)

    if alpha_facA < 0 and yellow_switch==True:
        with open(path_to_txt_yellows,"a") as yellow_txt:
            yellow_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, parameters])+"\n")
    elif trackA < 0 and blue_switch==True:
        with open(path_to_txt_blues,"a") as blue_txt:
            blue_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, parameters])+"\n")
    elif early_Omega_DE_max > Omega_DE_arrA[0]: #less harsh early DE criterion, only checks up until redshift = early_DE_threshold
        with open(path_to_txt_pinks,"a") as pink_txt:
            pink_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, parameters, early_Omega_DE_max])+"\n")
    elif Omega_DE_max > Omega_DE_arrA[0] and red_switch==True:
        with open(path_to_txt_reds,"a") as red_txt:
            red_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, parameters, Omega_DE_max])+"\n")
    elif Omega_m_max < Omega_m_crit:
        with open(path_to_txt_magentas,"a") as magenta_txt:
            magenta_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, parameters])+"\n")
    elif ( (Omega_m_max > 1.0 + tolerance) or (Omega_m_min < 0.-tolerance) or
          (Omega_r_max > 1.0+ tolerance) or (Omega_r_min < 0.-tolerance) or
          (Omega_L_max > 1.0+ tolerance) or (Omega_L_min < 0.-tolerance) or
          (Omega_phi_max > 1.0+ tolerance) or (Omega_phi_min < 0.-tolerance) or
          (Omega_DE_max > 1.0+ tolerance) or (Omega_DE_min < 0.-tolerance)  ):
        with open(path_to_txt_blacks,"a") as black_txt:
            black_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, parameters])+"\n")
    elif ( (np.isnan(Omega_m_max)) or
          (np.isnan(Omega_r_max)) or
          (np.isnan(Omega_L_max)) or
          (np.isnan(Omega_phi_max)) or
          (np.isnan(Omega_DE_max))            ):
        with open(path_to_txt_greys,"a") as grey_txt:
            grey_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, parameters])+"\n")
    else:
        with open(path_to_txt_greens,"a") as green_txt:
            green_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, parameters])+"\n")

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
scan_metadata_file = open(path_to_metadata, "a")
print('--- Runtime information ---',file=scan_metadata_file)
print("Script duration: "+str(end-start),file=scan_metadata_file)

scan_metadata_file.close()
