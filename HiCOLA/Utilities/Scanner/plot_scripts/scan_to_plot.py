#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 02:47:29 2023

@author: ashimsg
"""
import HiCOLA.Frontend.expression_builder as eb
import HiCOLA.Utilities.Other.alpha_and_sound_speed as al
import HiCOLA.Frontend.numerical_solver as ns
import sympy as sym
from matplotlib import pyplot as plt
import numpy as np

to_exec = eb.declare_symbols()
exec(to_exec)
Xreal = 0.5*(E**2.)*phiprime**2.

k1, k2, g31, g32= sym.symbols('k_1 k_2 g_{31} g_{32}')
symbol_list = [k1, k2, g31, g32]
odeint_parameter_symbols = [E, phi, phiprime, omegar, omegam]
closure_declaration = ['odeint_parameters',2]

def scan_to_background(G3, G4, K,  symbol_list, mass_ratio_list, Hubble0, Scalar0, Scalarprime0,Omega_r0, Omega_m0 , Omega_l0,fsolve_ier, parameters ):
    
    read_out_dict = {}

    simulation_parameters = [1000, 1000., False, 0., False]
    
    read_out_dict.update({'simulation_parameters':simulation_parameters, 'symbol_list':symbol_list, 'odeint_parameter_symbols':odeint_parameter_symbols, 'closure_declaration':closure_declaration})

    lambdified_functions = eb.create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list)
    E_prime_E_lambda = lambdified_functions['E_prime_E_lambda']
    B2_lambda = lambdified_functions['B2_lambda']

    read_out_dict.update(lambdified_functions)

    cosmological_parameters = [Omega_r0, Omega_m0, Omega_l0]
    initial_conditions = [Hubble0, Scalar0, Scalarprime0]
    repack_dict = {'cosmological_parameters':cosmological_parameters, 'initial_conditions':initial_conditions, 'Horndeski_parameters':parameters}
    read_out_dict.update(repack_dict)

    background_quantities = ns.run_solver(read_out_dict)
    return background_quantities, lambdified_functions

M_pG4_test, M_KG4_test, M_G3s_test, M_sG4_test, M_G3G4_test, M_Ks_test, M_gp_test = 1., 1., 1., 1., 1., 1., 1. #maybe scale everything to M_p instead   
mass_ratio_list =[M_pG4_test, M_KG4_test, M_G3s_test, M_sG4_test, M_G3G4_test, M_Ks_test, M_gp_test]

Kt = k1*X + k2*X*X
G3t = g31*X + g32*X*X
G4t = 0.5

# bg_quant = scan_to_background(G3t, G4t,Kt,symbol_list, mass_ratio_list,
#                                     1.5580736543909348, 1.0, 0.89859472, 8.016144815327033e-05, 0.2797409788825341, 0.17460662965043744, 1, [-2.8439678244598756, -1.2255772945186205, -26.501799616513757, 13.929157328086628])

# bg_quant = scan_to_background(G3t, G4t,Kt,symbol_list, mass_ratio_list,
#                                     1.5612679388109132, 1.0, 0.9137839688875196, 8.016144815327033e-05, 0.2797409788825341, 0.3783143642426145, 1.0, [-0.397440209844333, -0.13916304083570108, -3.5804999320100737, 1.8796838411183758])

bg_quant, lamb_funcs = scan_to_background(G3t, G4t,Kt,symbol_list, mass_ratio_list,
                                    1.0221992772328343, 1.0, 0.89, 8.016144815327033e-05, 0.2797409788825341, 0.6193538193156088, 1, [-0.24967084080663832, -0.7056850878859002, -0.0035284254394295014, 0.4134138473198232])

phiprime_arr = np.arange(-1,1,0.01)
phi_arr = np.ones(len(phiprime_arr))
E0 = np.ones(len(phiprime_arr))*1.5612679388109132
omega_l = np.ones(len(phiprime_arr))*0.3783143642426145
omega_r = np.ones(len(phiprime_arr))*8.016144815327033e-05
omega_m = np.ones(len(phiprime_arr))*0.2797409788825341#0.3
k1_arr  = np.ones(len(phiprime_arr))*-0.397440209844333#-0.11442716734931437
k2_arr = np.ones(len(phiprime_arr))*-0.13916304083570108#-0.29293354841424546
g31_arr = np.ones(len(phiprime_arr))*-3.5804999320100737#-1.9452618449383468
g32_arr = np.ones(len(phiprime_arr))*1.8796838411183758#1.0405243750964335

fried_plot = lamb_funcs['fried_RHS_lambda'](E0,phi_arr, phiprime_arr,omega_l, omega_m, omega_r, k1_arr, k2_arr, g31_arr, g32_arr)


fig, [ax1,ax2] = plt.subplots(figsize=(20,11),nrows=2, ncols=1)
ax1.set_title('ESS background - yellow band above')
ax1.loglog(bg_quant['a'], np.ones(len(bg_quant['a'])), color='black', linestyle='--')
ax1.loglog(bg_quant['a'], bg_quant['Hubble'], label='U (Hubble)')
ax1.loglog(bg_quant['a'],bg_quant['omega_m'], label=r'$\Omega_m$')
ax1.loglog(bg_quant['a'], bg_quant['omega_r'], label=r'$\Omega_r$')
ax1.loglog(bg_quant['a'], bg_quant['omega_l'], label=r'$\Omega_{\Lambda}$')
ax1.loglog(bg_quant['a'], bg_quant['omega_phi'], label=r'$\Omega_{\phi}$')
ax1.loglog(bg_quant['a'], bg_quant['scalar_prime'], label=r'$\phi \ prime$')

ax2.semilogx(bg_quant['a'],bg_quant['Hubble_prime'], label=r'U prime')

ax1.legend()
ax2.legend()
fig.show()

fig22, ax22 = plt.subplots()
ax22.plot(phiprime_arr, fried_plot, label='fried RHS')
ax22.plot(phiprime_arr, np.ones(len(phiprime_arr)), linestyle='--')
ax22.legend()
fig22.show()

fig3, ax3 = plt.subplots()
ax3.semilogx(bg_quant['a'],bg_quant['omega_m'], label=r'$\Omega_m$')
ax3.semilogx(bg_quant['a'], bg_quant['omega_r'], label=r'$\Omega_r$')
ax3.semilogx(bg_quant['a'], bg_quant['omega_l'], label=r'$\Omega_{\Lambda}$')
ax3.semilogx(bg_quant['a'], bg_quant['omega_phi'], label=r'$\Omega_{\phi}$')
ax3.set_title('Model 3')
ax3.legend()
fig3.show()