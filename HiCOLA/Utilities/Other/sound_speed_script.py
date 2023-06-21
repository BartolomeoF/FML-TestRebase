#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 01:25:21 2023

@author: ashimsg
"""

import HiCOLA.Frontend.expression_builder as eb
import HiCOLA.Utilities.Other.alpha_and_sound_speed as al
import HiCOLA.Frontend.numerical_solver as ns
import sympy as sym
from matplotlib import pyplot as plt

to_exec = eb.declare_symbols()
exec(to_exec)
Xreal = 0.5*(E**2.)*phiprime**2.

k1, k2, g31, g32= sym.symbols('k_1 k_2 g_{31} g_{32}')
symbol_list = [k1, k2, g31, g32]
odeint_parameter_symbols = [E, phi, phiprime, omegar, omegam]
closure_declaration = ['odeint_parameters',2]

def sound_speed(G3, G4, K,  symbol_list, mass_ratio_list, Hubble0, Scalar0, Scalarprime0,Omega_r0, Omega_m0 , Omega_l0,fsolve_ier, parameters ):
    
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
    
    cs_sym = al.scalar_sound_speed(G3, G4, K,M_pG4 = 1,
    M_KG4 = 1,
    M_G3s = 1,
    M_sG4 = 1,
    M_G3G4 = 1,
    M_Ks = 1,)
    cs_sym = cs_sym.subs(X, Xreal)
    print(cs_sym)
    cs_lambda = sym.lambdify([E,Eprime,phi, phiprime,phiprimeprime, omegam, *symbol_list],cs_sym)
    print('!')
    print(cs_lambda(1,1,1,1,1,1,1,1,1,1))
    cs_arr = []
    for hv, hprimev, scalarv, scalarprimev, scalarprimeprimev, omegamv in zip( background_quantities['Hubble'], background_quantities['Hubble_prime'], background_quantities['scalar'],
                       background_quantities['scalar_prime'], background_quantities['scalar_primeprime'], background_quantities['omega_m']):
        cs_arr.append(cs_lambda(hv, hprimev, scalarv, scalarprimev, scalarprimeprimev, omegamv, *parameters))
    return cs_arr, background_quantities

# Hubble0          Scalar0          Scalarprime0          Omega_r0          Omega_m0          Omega_l0          fsolve_ier          k_1          k_2          g_31          g_32          

#[1.5580736543909348, 1.0, array([0.89859472]), 8.016144815327033e-05, 0.2797409788825341, 0.17460662965043744, 1, -2.8439678244598756, -1.2255772945186205, -26.501799616513757, 13.929157328086628]

M_pG4_test, M_KG4_test, M_G3s_test, M_sG4_test, M_G3G4_test, M_Ks_test, M_gp_test = 1., 1., 1., 1., 1., 1., 1. #maybe scale everything to M_p instead   
mass_ratio_list =[M_pG4_test, M_KG4_test, M_G3s_test, M_sG4_test, M_G3G4_test, M_Ks_test, M_gp_test]

Kt = k1*X + k2*X*X
G3t = g31*X + g32*X*X
G4t = 0.5

cs_arr_test, bg_quant = sound_speed(G3t, G4t,Kt,symbol_list, mass_ratio_list,
                                    1.5580736543909348, 1.0, 0.89859472, 8.016144815327033e-05, 0.2797409788825341, 0.17460662965043744, 1, [-2.8439678244598756, -1.2255772945186205, -26.501799616513757, 13.929157328086628])
print(cs_arr_test)

fig, ax = plt.subplots()
ax.semilogx(bg_quant['a'], cs_arr_test, label='ESS sound speed')
ax.legend()
fig.show()