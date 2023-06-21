#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 21:43:08 2023

@author: ashimsg
"""

from HiCOLA.Frontend import expression_builder as eb
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

to_exec = eb.declare_symbols()
exec(to_exec)

k1, k2, g31, g32 = sym.symbols('k_1 k_2 g_{31} g_{32}')




M_pG4_test = 1.
M_KG4_test = 1.
M_G3s_test = 1.
M_sG4_test = 1.
M_G3G4_test = 1.
M_Ks_test = 1.

K = k1*X + k2*X*X
G3 = g31*X + g32*X*X
G4 = 0.5

fried_RHS = eb.fried_closure(G3, G4,  K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test, M_G3G4=M_G3G4_test, M_Ks=M_Ks_test)
fried_RHS_lambda = sym.lambdify([E,phiprime,omegal,omegam,omegar,k1,k2,g31,g32],fried_RHS)


#[-0.397440209844333, -0.13916304083570108, -3.5804999320100737, 1.8796838411183758]

phiprime_arr = np.arange(-1,1,0.01)
E0 = np.ones(len(phiprime_arr))*1.5612679388109132
omega_l = np.ones(len(phiprime_arr))*0.3783143642426145
omega_r = np.ones(len(phiprime_arr))*8.016144815327033e-05
omega_m = np.ones(len(phiprime_arr))*0.2797409788825341#0.3
k1_arr  = np.ones(len(phiprime_arr))*-0.397440209844333#-0.11442716734931437
k2_arr = np.ones(len(phiprime_arr))*-0.13916304083570108#-0.29293354841424546
g31_arr = np.ones(len(phiprime_arr))*-3.5804999320100737#-1.9452618449383468
g32_arr = np.ones(len(phiprime_arr))*1.8796838411183758#1.0405243750964335


fried_plot = fried_RHS_lambda(E0,phiprime_arr,omega_l, omega_m, omega_r, k1_arr, k2_arr, g31_arr, g32_arr)
fried_sub_plot = fried_RHS_sub_lambda(E0, phiprime_arr, omega_l, omega_m, omega_r, k1_arr, g31_arr)
fig, ax = plt.subplots()
ax.plot(phiprime_arr, fried_plot,label='friedman RHS')
ax.plot(phiprime_arr, fried_sub_plot, label='friedmann cubic RHS')
ax.plot(phiprime_arr, np.zeros(len(phiprime_arr)),linestyle='--')
ax.set_xlabel('$\phi\'_0$')
ax.legend()
fig.show()

fried_RHS = fried_RHS.expand()
fried_RHS = fried_RHS.collect(phiprime)
coefficients = fried_RHS.collect(phiprime).as_coefficients_dict()

coeffdict = {phiprime**float(p): fried_RHS.coeff(phiprime**float(p)) for p in range(1,6)}
coefflambdict = {}
for key in coeffdict:
    lambcoeff = sym.lambdify([E,phiprime,omegal,omegam,omegar,k1,k2,g31,g32],coeffdict[key])
    coefflambdict.update({key:lambcoeff})
    

    
# E_arr = np.arange(1,100,1)
# phiprime_arr = np.ones(len(E_arr))
# Omega_m_arr = np.ones(len(E_arr))*0.3
# Omega_l_arr = np.zeros(len(E_arr))
# Omega_r_arr = Omega_l_arr

# k1_arr  = np.ones(len(phiprime_arr))*-0.11442716734931437
# k2_arr = np.ones(len(phiprime_arr))*-0.29293354841424546
# g31_arr = np.ones(len(phiprime_arr))*-1.9452618449383468
# g32_arr = np.ones(len(phiprime_arr))*1.0405243750964335

# cubic_arr = coefflambdict[phiprime**3.0](E_arr, phiprime_arr, Omega_l_arr, Omega_m_arr, Omega_r_arr, k1_arr, k2_arr, g31_arr, g32_arr)
