#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:36:34 2023

@author: ashimsg
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from HiCOLA.Utilities.Other.support import compute_fphi, ESS_direct_to_seed
from HiCOLA.Frontend.read_parameters import read_scan_result

directory = '/home/ashimsg/Documents/QMUL_Desktop/Horndeski_COLA/Hi-COLA/Output/Scanner/from_apocrita/2023-06-02/14-42_05/'

colours = ['green', 'grey', 'black', 'magenta', 'pink', 'red', 'blue', 'yellow']

files = glob.glob(directory+'*.txt')

#collect only filenames that pertain to scan results
results = {}
for colour in colours:
    for file in files:
        if colour in file:
            results.update({colour:file})
            
green_results = {}
grey_results = {}
black_results = {}
magenta_results = {}
pink_results = {}
red_results = {}
blue_results = {}
yellow_results = {}

for key, result in results.items():
    
    if key == 'green':
        result_dict = read_scan_result(results[key])
        green_results.update(result_dict)
    # if key == 'grey':
    #     result_dict = read_scan_result(results[key])
    #     grey_results.update(result_dict)
    # if key == 'black':
    #     result_dict = read_scan_result(results[key])
    #     black_results.update(result_dict)
    # if key == 'magenta':
    #     result_dict = read_scan_result(results[key])
    #     magenta_results.update(result_dict)
    if key == 'pink':
        result_dict = read_scan_result(results[key])
        pink_results.update(result_dict)
    # if key == 'red':
    #     result_dict = read_scan_result(results[key])
    #     red_results.update(result_dict)
    # if key == 'blue':
    #     result_dict = read_scan_result(results[key])
    #     blue_results.update(result_dict)
    # if key == 'yellow':
    #     result_dict = read_scan_result(results[key])
    #     yellow_results.update(result_dict)
            
        
green_EdS = [1/U0 for U0 in green_results['U0_arr'] ]
green_f = [compute_fphi(omega_l, omega_m, omega_r) for omega_l, omega_m, omega_r in zip(green_results['Omega_l0_arr'], green_results['Omega_m0_arr'], green_results['Omega_r0_arr']  )]
green_k1seed = [ESS_direct_to_seed(k1, g31, omega_l0, f_phi, EdS)[0] for k1, g31, omega_l0, f_phi, EdS in zip(green_results['k1_arr'], green_results['g31_arr'], green_results['Omega_l0_arr'], green_f, green_EdS )]
green_g31seed = [ESS_direct_to_seed(k1, g31, omega_l0, f_phi, EdS)[1] for k1, g31, omega_l0, f_phi, EdS in zip(green_results['k1_arr'], green_results['g31_arr'], green_results['Omega_l0_arr'], green_f, green_EdS )]
# grey_EdS = [1/U0 for U0 in grey_results['U0_arr'] ]
# grey_f = [compute_fphi(omega_l, omega_m, omega_r) for omega_l, omega_m, omega_r in zip(grey_results['Omega_l0_arr'], grey_results['Omega_m0_arr'], grey_results['Omega_r0_arr']  )]
# grey_k1seed = [ESS_direct_to_seed(k1, g31, omega_l0, f_phi, EdS)[0] for k1, g31, omega_l0, f_phi, EdS in zip(grey_results['k1_arr'], grey_results['g31_arr'], grey_results['Omega_l0_arr'], grey_f, grey_EdS )]
# black_EdS = [1/U0 for U0 in black_results['U0_arr'] ]
# black_f = [compute_fphi(omega_l, omega_m, omega_r) for omega_l, omega_m, omega_r in zip(black_results['Omega_l0_arr'], black_results['Omega_m0_arr'], black_results['Omega_r0_arr']  )]
# black_k1seed = [ESS_direct_to_seed(k1, g31, omega_l0, f_phi, EdS)[0] for k1, g31, omega_l0, f_phi, EdS in zip(black_results['k1_arr'], black_results['g31_arr'], black_results['Omega_l0_arr'], black_f, black_EdS )]
pink_EdS = [1/U0 for U0 in pink_results['U0_arr'] ]
pink_f = [compute_fphi(omega_l, omega_m, omega_r) for omega_l, omega_m, omega_r in zip(pink_results['Omega_l0_arr'], pink_results['Omega_m0_arr'], pink_results['Omega_r0_arr']  )]
pink_k1seed = [ESS_direct_to_seed(k1, g31, omega_l0, f_phi, EdS)[0] for k1, g31, omega_l0, f_phi, EdS in zip(pink_results['k1_arr'], pink_results['g31_arr'], pink_results['Omega_l0_arr'], pink_f, pink_EdS )]
pink_g31seed = [ESS_direct_to_seed(k1, g31, omega_l0, f_phi, EdS)[1] for k1, g31, omega_l0, f_phi, EdS in zip(pink_results['k1_arr'], pink_results['g31_arr'], pink_results['Omega_l0_arr'], pink_f, pink_EdS )]
# red_EdS = [1/U0 for U0 in red_results['U0_arr'] ]
# red_f = [compute_fphi(omega_l, omega_m, omega_r) for omega_l, omega_m, omega_r in zip(red_results['Omega_l0_arr'], red_results['Omega_m0_arr'], red_results['Omega_r0_arr']  )]
# red_k1seed = [ESS_direct_to_seed(k1, g31, omega_l0, f_phi, EdS)[0] for k1, g31, omega_l0, f_phi, EdS in zip(red_results['k1_arr'], red_results['g31_arr'], red_results['Omega_l0_arr'], red_f, red_EdS )]
# magenta_EdS = [1/U0 for U0 in magenta_results['U0_arr'] ]
# magenta_f = [compute_fphi(omega_l, omega_m, omega_r) for omega_l, omega_m, omega_r in zip(magenta_results['Omega_l0_arr'], magenta_results['Omega_m0_arr'], magenta_results['Omega_r0_arr']  )]
# magenta_k1seed = [ESS_direct_to_seed(k1, g31, omega_l0, f_phi, EdS)[0] for k1, g31, omega_l0, f_phi, EdS in zip(magenta_results['k1_arr'], magenta_results['g31_arr'], magenta_results['Omega_l0_arr'], magenta_f, magenta_EdS )]
# blue_EdS = [1/U0 for U0 in blue_results['U0_arr'] ]
# blue_f = [compute_fphi(omega_l, omega_m, omega_r) for omega_l, omega_m, omega_r in zip(blue_results['Omega_l0_arr'], blue_results['Omega_m0_arr'], blue_results['Omega_r0_arr']  )]
# blue_k1seed = [ESS_direct_to_seed(k1, g31, omega_l0, f_phi, EdS)[0] for k1, g31, omega_l0, f_phi, EdS in zip(blue_results['k1_arr'], blue_results['g31_arr'], blue_results['Omega_l0_arr'], blue_f, blue_EdS )]
# yellow_EdS = [1/U0 for U0 in yellow_results['U0_arr'] ]
# yellow_f = [compute_fphi(omega_l, omega_m, omega_r) for omega_l, omega_m, omega_r in zip(yellow_results['Omega_l0_arr'], yellow_results['Omega_m0_arr'], yellow_results['Omega_r0_arr']  )]
# yellow_k1seed = [ESS_direct_to_seed(k1, g31, omega_l0, f_phi, EdS)[0] for k1, g31, omega_l0, f_phi, EdS in zip(yellow_results['k1_arr'], yellow_results['g31_arr'], yellow_results['Omega_l0_arr'], yellow_f, yellow_EdS )]

# total_no = 0
# for i in [green_EdS, grey_EdS, black_EdS, pink_EdS, magenta_EdS, blue_EdS, yellow_EdS]:
#     total_no += len(i)
# print(f'number of models saved = {total_no}')

fig, ax = plt.subplots(figsize=(20,11))
#y b p r m gre blk green



#ax.scatter(magenta_f, magenta_EdS, c='magenta')

# ax.scatter(black_f, black_EdS, c='black')
# ax.scatter(pink_f, pink_EdS, c='pink')
# ax.scatter(grey_f, grey_EdS, c='black')
# ax.scatter(green_f, green_EdS, c='green')
# ax.plot(pink_f, pink_EdS, color='pink', marker='.')
# ax.plot(magenta_f, magenta_EdS, color='magenta', marker='.')

# ax.plot(black_f, black_EdS, color='black', marker='.')
# ax.plot(green_f, green_EdS, 'g.')
# ax.plot(grey_f, grey_EdS, color='grey', marker='.')

# ax.scatter(pink_EdS, pink_k1seed, color='pink')
EdS_points = np.concatenate((green_EdS, pink_EdS))
k1seed_points = np.concatenate((green_k1seed, pink_k1seed))
#phiprime0_vals = np.concatenate((green_results['phiprime0_arr'], pink_results['phiprime0_arr']))
f_phi_vals = np.concatenate((green_f, pink_f))
#scatter = ax.scatter(green_EdS, green_k1seed, c=green_results['phiprime0_arr'], cmap='viridis',s=1)
#scatter_pink = ax.scatter(pink_EdS, pink_k1seed, c=pink_results['phiptime0_arr'], cmap='viridis', marker='v', s=1)
#scatter = ax.scatter(EdS_points, k1seed_points, c=list(f_phi_vals), cmap='viridis',vmax = np.max(green_results['phiprime0_arr']), vmin = np.min(green_results['phiprime0_arr']),s=8)
scatter = ax.scatter(EdS_points, k1seed_points, c=list(f_phi_vals), cmap='viridis',vmax = np.max(green_f), vmin = np.min(green_f),s=8)
#cbar = plt.colorbar(scatter, label=r'$\phi^{\prime}_{0}$')
cbar = plt.colorbar(scatter, label=r'$f_{\phi}$')

ax.set_xlabel('$E_{dS}$')
ax.set_ylabel('$k_{1-seed}$')
ax.set_title('ESS Slice')
fig.show()



fig2, [ax21, ax22] = plt.subplots(nrows=1, ncols=2)
greenphi0counts, greenbins, greenpatches = plt.hist(green_results['phiprime0_arr'],bins=10)
pinkphi0counts, pinkbins, pinkpatches = plt.hist(pink_results['phiprime0_arr'],bins=10)
ax21.hist(greenbins[:-1], greenbins, weights=greenphi0counts)
ax21.set_title('Green $\phi\'_0$ counts')
ax22.hist(pinkbins[:-1], pinkbins, weights=pinkphi0counts)
ax22.set_title('Pink $\phi\'_0$ counts')
# fig2, ax2 = plt.subplots()
# ax2.plot(np.arange(0, len(green_g31seed)), green_g31seed)
# fig2.show()
