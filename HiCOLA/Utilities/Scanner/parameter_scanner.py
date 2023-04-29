import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import HiCOLA.Frontend.numerical_solver as ns
import HiCOLA.Frontend.expression_builder as eb
import os
import time
import itertools as it
from multiprocessing import Pool
from HiCOLA.Frontend.read_parameters import read_in_scan_parameters, read_in_scan_settings
from argparse import ArgumentParser
from configobj import ConfigObj
from pathlib import Path
import shutil

##############
symbol_decl = eb.declare_symbols()
exec(symbol_decl)
odeint_parameter_symbols = [E, phiprime, omegar, omegam]
######################


start = time.time()
##############

parser = ArgumentParser(prog='Scanner')

parser.add_argument('input_ini_filenames',nargs=2)
# parser.add_argument('-d', '--direct',action='store_true')

args = parser.parse_args()
# print(args)
filenames = args.input_ini_filenames
scan_settings_path = filenames[0]
scan_values_path = filenames[1]


scan_ini_path = Path(scan_settings_path).resolve()
scan_param_path = Path(scan_values_path)

scan_settings_dict = read_in_scan_settings(scan_settings_path)
scan_settings_dict.update({'odeint_parameter_symbols':odeint_parameter_symbols})

N_proc = scan_settings_dict['processor_number']

read_scan_values_from_file = scan_settings_dict['scan_values_from_file']

#[U0, phiprime0, Omega_r0, Omega_m0, Omega_l0, [k1dS, k2dS, g31dS, g32dS] ]
if read_scan_values_from_file is True:
     protoscan_list = np.loadtxt(scan_values_path, unpack=True)
     # print(protoscan_list)
     # print(protoscan_list.shape)
     # print('Printing scan_values_dict')
     # print(protoscan_list)
     # print(len(protoscan_list))
     # print('first entry')
     # print(protoscan_list[0])
     # print(type(protoscan_list[0]))
     zipped_protoscan = protoscan_list.transpose()
     #print(zipped_protoscan)
     # print(zipped_protoscan.shape)
     number_of_horndeski_parameters = len(zipped_protoscan[0][6:])
     scan_list = []
     for i in zipped_protoscan:
        U0, phi0, phiprime0, Omegar0, Omegam0, Omegal0 = i[:6] 
        horndeski_parameters = i[6:] #this packs parameters into a list
        j = [U0, phi0, phiprime0, Omegar0, Omegam0, Omegal0, horndeski_parameters, scan_settings_dict]
        scan_list.append(j)
     print('scan list')
     for l in scan_list:
        print(l)
        print('\n')
     scan_list_length = len(scan_list)
     print(f'scan list length = {scan_list_length}')
    #[U0_array, phiprime0_array, Omega_r0_array, Omega_m0_array, Omega_l0_array, parameter_arrays] = scan_values_dict
else:
    scan_values_dict = read_in_scan_parameters(scan_values_path)
    [U0_array, phi0_array, phiprime0_array] = scan_values_dict['initial_condition_arrays']
    [Omega_r0_array, Omega_m0_array, Omega_l0_array] = scan_values_dict['cosmological_parameter_arrays']
    parameter_arrays = scan_values_dict('Horndeski_parameter_arrays')
    number_of_horndeski_parameters = len(parameter_arrays)
    
    scan_list = it.product(U0_array, phi0_array, phiprime0_array, Omega_r0_array,Omega_m0_array,Omega_l0_array,*parameter_arrays, [scan_settings_dict])
    # parameter_cartesian_product = it.product(*parameter_arrays)
    scan_list2 = it.product(U0_array, phi0_array, phiprime0_array, Omega_r0_array,Omega_m0_array,Omega_l0_array,*parameter_arrays, [scan_settings_dict])
    scan_list_to_print = list(scan_list2)
    print(len(scan_list_to_print))
    # for i in scan_list_to_print:
    #     print(i)



model = scan_settings_dict['model_name']

##################################################

description_string = 'Parallelised parameter scanner using pool.starmap on a functional form of scanning - PLL. No list on iter.prod. processes = '+str(N_proc)


##To include date in log file names
time_snap = time.localtime()
folder_date = time.strftime("%Y-%m-%d",time_snap)
folder_time = time.strftime("%H-%M_%S", time_snap)
file_date = time.strftime("%Y-%m-%d_%H-%M-%S",time_snap)

##To create a directory to store logs if needed
outputdir = scan_settings_dict['output_directory']
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

##Sub-directory to organise files by date of creation

saving_directory =outputdir+folder_date+"/"

##Sub-directory to organise multiple runs performed on the same day by time of creation
saving_subdir = saving_directory+"/"+folder_time+"/"

if not os.path.exists(saving_directory):
    os.makedirs(saving_directory)
if not os.path.exists(saving_subdir):
    os.makedirs(saving_subdir)

# copy the .ini files used for the run into the output directory (can  be  re-used  as input files for subsequent run)
shutil.copy2(scan_ini_path, saving_subdir+file_date+'_'+model+'_scan_settings.ini')
if read_scan_values_from_file is False:
    shutil.copy2(scan_param_path, saving_subdir+file_date+'_'+model+'_scan_values.ini')

path_to_txt_greens = saving_subdir+file_date+'_'+model+"_greens.txt"
path_to_txt_greys = saving_subdir+file_date+'_'+model+"_greys.txt"
path_to_txt_blacks = saving_subdir+file_date+'_'+model+"_blacks.txt"
path_to_txt_magentas = saving_subdir+file_date+'_'+model+"_magentas.txt"
path_to_txt_pinks = saving_subdir+file_date+'_'+model+"_pinks.txt"
path_to_txt_reds = saving_subdir+file_date+'_'+model+"_reds.txt"
path_to_txt_blues = saving_subdir+file_date+'_'+model+"_blues.txt"
path_to_txt_yellows = saving_subdir+file_date+'_'+model+"_yellows.txt"


early_DE_z = scan_settings_dict['early_DE_threshold']
horndeski_parameter_symbols = scan_settings_dict['symbol_list']
for i in [path_to_txt_greens, path_to_txt_greys, path_to_txt_blacks, path_to_txt_magentas, 
          path_to_txt_pinks, path_to_txt_reds, path_to_txt_blues, path_to_txt_yellows]:
    scan_file = open(i,"a")
    spaces = '          '
    print('# Hubble0'+spaces+'Scalarprime0'+spaces+'Omega_r0'+spaces+'Omega_m0'+spaces+'Omega_l0',end=spaces,file=scan_file)
    for j in np.arange(number_of_horndeski_parameters):
        print(horndeski_parameter_symbols[j],end=spaces, file=scan_file)
    if 'red' in i:
        print('max[Omega_DE]', end=spaces, file=scan_file)
    if 'pink' in i:
    	    print(f'max[Omega_DE(z>{early_DE_z})]', end=spaces, file=scan_file)
    print('\n',file=scan_file)


###Metadata file
#potential redundancy with .ini files that are now copied over into output directory
path_to_metadata = saving_subdir+file_date+'_'+model+"_scan_metadata.txt"
scan_settings_dict_copy = scan_settings_dict.copy()
#print(scan_settings_dict_copy)
write_dict = ConfigObj(scan_settings_dict_copy)
#print(write_dict)
write_dict.filename = path_to_metadata
write_dict.write()

def parameter_scanner(U0, phi0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, parameters, settings_dict):
    
    read_out_dict = {}

    read_out_dict = settings_dict.copy()
    red_switch = settings_dict['red_switch']
    blue_switch = settings_dict['blue_switch']
    yellow_switch = settings_dict['yellow_switch']
    tolerance = settings_dict['tolerance']
    early_DE_threshold = settings_dict['early_DE_threshold']
    Omega_m_crit = settings_dict['Omega_m_crit']

    K = settings_dict['K']
    G3 = settings_dict['G3']
    G4 = settings_dict['G4']
    mass_ratio_list = settings_dict['mass_ratio_list']
    symbol_list = settings_dict['symbol_list']

    lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list)
    E_prime_E_lambda = lambdified_functions['E_prime_E_lambda']
    B2_lambda = lambdified_functions['B2_lambda']

    read_out_dict.update(lambdified_functions)



    cosmological_parameters = [Omega_r0, Omega_m0, Omega_l0]
    initial_conditions = [U0, phi0, phi_prime0]
    repack_dict = {'cosmological_parameters':cosmological_parameters, 'initial_conditions':initial_conditions, 'Horndeski_parameters':parameters}
    read_out_dict.update(repack_dict)


    

    background_quantities = ns.run_solver(read_out_dict)

    Omega_L_arrA = background_quantities['omega_l']
    Omega_r_arrA = background_quantities['omega_r']
    Omega_m_arrA = background_quantities['omega_m']
    Omega_phi_arrA = background_quantities['omega_phi']
    Omega_DE_arrA = background_quantities['omega_DE']
    a_arr_invA = background_quantities['a']
    U_arrA = background_quantities['Hubble']
    y_arrA = background_quantities['scalar_prime']
    Iy_arrA = background_quantities['scalar']

    alpha_facA = 1.-Omega_L_arrA[0]*U0*U0

    trackA = B2_lambda(U_arrA[0],Iy_arrA[0], y_arrA[0],*parameters) #general version
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
            yellow_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, *parameters])+"\n")
    elif trackA < 0 and blue_switch==True:
        with open(path_to_txt_blues,"a") as blue_txt:
            blue_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, *parameters])+"\n")
    elif early_Omega_DE_max > Omega_DE_arrA[0]: #less harsh early DE criterion, only checks up until redshift = early_DE_threshold
        with open(path_to_txt_pinks,"a") as pink_txt:
            pink_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, *parameters, early_Omega_DE_max])+"\n")
    elif Omega_DE_max > Omega_DE_arrA[0] and red_switch==True:
        with open(path_to_txt_reds,"a") as red_txt:
            red_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, *parameters, Omega_DE_max])+"\n")
    elif Omega_m_max < Omega_m_crit:
        with open(path_to_txt_magentas,"a") as magenta_txt:
            magenta_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, *parameters])+"\n")
    elif ( (Omega_m_max > 1.0 + tolerance) or (Omega_m_min < 0.-tolerance) or
          (Omega_r_max > 1.0+ tolerance) or (Omega_r_min < 0.-tolerance) or
          (Omega_L_max > 1.0+ tolerance) or (Omega_L_min < 0.-tolerance) or
          (Omega_phi_max > 1.0+ tolerance) or (Omega_phi_min < 0.-tolerance) or
          (Omega_DE_max > 1.0+ tolerance) or (Omega_DE_min < 0.-tolerance)  ):
        with open(path_to_txt_blacks,"a") as black_txt:
            black_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, *parameters])+"\n")
    elif ( (np.isnan(Omega_m_max)) or
          (np.isnan(Omega_r_max)) or
          (np.isnan(Omega_L_max)) or
          (np.isnan(Omega_phi_max)) or
          (np.isnan(Omega_DE_max))            ):
        with open(path_to_txt_greys,"a") as grey_txt:
            grey_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, *parameters])+"\n")
    else:
        with open(path_to_txt_greens,"a") as green_txt:
            green_txt.write(str([U0, phi_prime0, Omega_r0, Omega_m0,Omega_l0, *parameters])+"\n")
    

if __name__ == '__main__':
    pool = Pool(processes=N_proc)
    pool.starmap(parameter_scanner, scan_list)


end = time.time()
scan_metadata_file = open(path_to_metadata, "a")
print('--- Runtime information ---',file=scan_metadata_file)
print("Script duration: "+str(end-start),file=scan_metadata_file)

scan_metadata_file.close()
