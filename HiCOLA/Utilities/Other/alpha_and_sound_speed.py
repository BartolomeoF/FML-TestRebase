#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:45:33 2022

@author: ashimsg
"""

import HiCOLA.Frontend.expression_builder as eb
import numpy as np
import sympy as sym


def Xprime(E='E',
        Eprime='Eprime',
        phiprime='phiprime',
        phiprimeprime='phiprimeprime',
        X='X'):
    parameters = [E, Eprime, phiprime, phiprimeprime, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = eb.sy(parameters[i])
    [E, Eprime, phiprime, phiprimeprime, X] = parameters
    Xprime = E*Eprime*phiprime*phiprime + E*E*phiprime*phiprimeprime
    return Xprime

def Mstar_squared(G3, G4,  K, 
        E='E',
        Eprime='Eprime',
        M_pG4 = 'M_{pG4}',
        M_KG4 = 'M_{KG4}',
        M_G3s = 'M_{G3s}',
        M_sG4 = 'M_{sG4}',
        M_G3G4 = 'M_{G3G4}',
        M_Ks = 'M_{Ks}',
        phi='phi',
        phiprime='phiprime',
        phiprimeprime='phiprimeprime',
        omegar='Omega_r',
        omegam='Omega_m',
        omegal = 'Omega_l',
        X='X'): #This is dimensionless, to get dimensionful version as in https://arxiv.org/pdf/1404.3713.pdf, multiply by M_G4^2
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = eb.G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = eb.G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = eb.K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = eb.sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X] = parameters
    Mstar_sq = 2.*(G4 - 2.*X*G4x)
    return Mstar_sq

def alpha_M(G3, G4,  K, 
        E='E',
        Eprime='Eprime',
        M_pG4 = 'M_{pG4}',
        M_KG4 = 'M_{KG4}',
        M_G3s = 'M_{G3s}',
        M_sG4 = 'M_{sG4}',
        M_G3G4 = 'M_{G3G4}',
        M_Ks = 'M_{Ks}',
        phi='phi',
        phiprime='phiprime',
        phiprimeprime='phiprimeprime',
        omegar='Omega_r',
        omegam='Omega_m',
        omegal = 'Omega_l',
        X='X'): 
    print(G4)
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = eb.G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = eb.G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = eb.K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = eb.sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X] = parameters
    Xpr = Xprime()
    print(G4)
    print(G4phi)
    print(G4x)
    numerator = 2.*(G4phi*phiprime + G4x*Xpr + 2.*( Xpr*G4x + X*G4xx*Xpr + X*G4xphi*phiprime )  )
    print(sym.latex(numerator))
    denominator = Mstar_squared(G3,G4,K)
    alpha_M = sym.sqrt( ( 2.*numerator)/denominator)
    return alpha_M

def alpha_K(G3, G4,  K, 
        E='E',
        Eprime='Eprime',
        M_pG4 = 'M_{pG4}',
        M_KG4 = 'M_{KG4}',
        M_G3s = 'M_{G3s}',
        M_sG4 = 'M_{sG4}',
        M_G3G4 = 'M_{G3G4}',
        M_Ks = 'M_{Ks}',
        phi='phi',
        phiprime='phiprime',
        phiprimeprime='phiprimeprime',
        omegar='Omega_r',
        omegam='Omega_m',
        omegal = 'Omega_l',
        X='X'): 
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = eb.G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = eb.G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = eb.K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = eb.sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X] = parameters
    G4phixx = sym.diff(G4phix,X)
    Mstar_sq = Mstar_squared(G3,G4,K, M_pG4=M_pG4,M_KG4=M_KG4,M_G3s=M_G3s,M_sG4=M_sG4,M_G3G4=M_G3G4,M_Ks=M_Ks)
    term1 = 2.*M_sG4*X*(M_Ks*M_Ks*Kx + 2.*M_Ks*M_Ks*X*Kxx - 2.*M_G3s*G3phi - 2.*M_G3s*X*G3phix)
    term2 = 12.*phiprime*X*E*E*(M_G3s*G3x + M_G3s*X*G3xx - 3*(1./M_sG4)*(1./M_sG4)*G4phix - 2.*(1./M_sG4)*(1./M_sG4)*X*G4phixx)
    term3 = 12.*X*E*E*( (1./M_sG4)*(1./M_sG4)*G4x + 8.*(1./M_sG4)*(1./M_sG4)*X*G4xx + 4.*(1./M_sG4)*(1./M_sG4)*X*X*G4xx )
    alphaK = (term1 + term2 + term3)/( E*E*Mstar_sq )
    return alphaK

def alpha_B(G3, G4,  K, 
        E='E',
        Eprime='Eprime',
        M_pG4 = 'M_{pG4}',
        M_KG4 = 'M_{KG4}',
        M_G3s = 'M_{G3s}',
        M_sG4 = 'M_{sG4}',
        M_G3G4 = 'M_{G3G4}',
        M_Ks = 'M_{Ks}',
        phi='phi',
        phiprime='phiprime',
        phiprimeprime='phiprimeprime',
        omegar='Omega_r',
        omegam='Omega_m',
        omegal = 'Omega_l',
        X='X'): 
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = eb.G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = eb.G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = eb.K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = eb.sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X] = parameters
    Mstar_sq = Mstar_squared(G3,G4,K, M_pG4=M_pG4,M_KG4=M_KG4,M_G3s=M_G3s,M_sG4=M_sG4,M_G3G4=M_G3G4,M_Ks=M_Ks)
    term1 = 2.*M_sG4*E*phiprime*(M_G3G4*X*G3x - (1./M_sG4)*G4phi - (1./M_sG4)*X*G4phix)
    term2 = 8.*X*E*(G4x + X*G4xx)
    alphaB = (term1 + term2)/( E*Mstar_sq )
    return alphaB

def alpha_T(G3, G4,  K, 
        E='E',
        Eprime='Eprime',
        M_pG4 = 'M_{pG4}',
        M_KG4 = 'M_{KG4}',
        M_G3s = 'M_{G3s}',
        M_sG4 = 'M_{sG4}',
        M_G3G4 = 'M_{G3G4}',
        M_Ks = 'M_{Ks}',
        phi='phi',
        phiprime='phiprime',
        phiprimeprime='phiprimeprime',
        omegar='Omega_r',
        omegam='Omega_m',
        omegal = 'Omega_l',
        X='X'): 
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = eb.G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = eb.G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = eb.K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = eb.sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X] = parameters
    Mstar_sq = Mstar_squared(G3,G4,K, M_pG4=M_pG4,M_KG4=M_KG4,M_G3s=M_G3s,M_sG4=M_sG4,M_G3G4=M_G3G4,M_Ks=M_Ks)
    alphaT = 4.*X*G4x / Mstar_sq
    return alphaT

def alpha_B_prime(G3, G4,  K, 
        E='E',
        Eprime='Eprime',
        M_pG4 = 'M_{pG4}',
        M_KG4 = 'M_{KG4}',
        M_G3s = 'M_{G3s}',
        M_sG4 = 'M_{sG4}',
        M_G3G4 = 'M_{G3G4}',
        M_Ks = 'M_{Ks}',
        phi='phi',
        phiprime='phiprime',
        phiprimeprime='phiprimeprime',
        omegar='Omega_r',
        omegam='Omega_m',
        omegal = 'Omega_l',
        X='X'): 
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = eb.G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = eb.G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = eb.K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = eb.sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X] = parameters
    Xpr = Xprime()
    G4xxx = sym.diff(G4xx,X)
    G4xxphi = sym.diff(G4xx,phi)
    G4phixx = sym.diff(G4phix,X)
    G4phixphi = sym.diff(G4phix,phi)
    Mstar_sq = Mstar_squared(G3,G4,K, M_pG4=M_pG4,M_KG4=M_KG4,M_G3s=M_G3s,M_sG4=M_sG4,M_G3G4=M_G3G4,M_Ks=M_Ks)
    alphab = alpha_B(G3,G4,K, M_pG4=M_pG4,M_KG4=M_KG4,M_G3s=M_G3s,M_sG4=M_sG4,M_G3G4=M_G3G4,M_Ks=M_Ks)
    bracket1 = (M_G3G4*X*G3x - (1./M_sG4)*G4phi - (1./M_sG4)*X*G4phix)
    term1 = 2.*M_sG4*bracket1*(Eprime*phiprime + E*phiprimeprime)
    term21 =  M_G3G4*Xpr*G3x + M_G3G4*X*(G3xphi*phiprime + G3xx*Xpr)  
    term22 = (-1./M_sG4)*(G4phiphi*phiprime + G4phix*Xpr) - (1./M_sG4)*Xpr*G4phix - (1./M_sG4)*X*(G4phixx + G4phixphi*phiprime)
    term2 = 2.*M_sG4*E*phiprime*(term21 + term22)
    terma = term1 + term2
    term3 = 8.*(G4x + X*G4xx)*(Xpr*E + X*Eprime)
    term4 = 8.*X*E*( G4xphi*phiprime + 2.*G4xx*Xpr + X*( G4xxx*Xpr + G4xxphi*phiprime ) )
    R = term1 + term2 + term3 + term4
    alphabprime = R/(E*Mstar_sq) - (Eprime/E)*alphab
    return alphabprime

def scalar_sound_speed(G3, G4,  K, 
        E='E',
        Eprime='Eprime',
        M_pG4 = 'M_{pG4}',
        M_KG4 = 'M_{KG4}',
        M_G3s = 'M_{G3s}',
        M_sG4 = 'M_{sG4}',
        M_G3G4 = 'M_{G3G4}',
        M_Ks = 'M_{Ks}',
        phi='phi',
        phiprime='phiprime',
        phiprimeprime='phiprimeprime',
        omegar='Omega_r',
        omegam='Omega_m',
        omegal = 'Omega_l',
        X='X'): 
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = eb.G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = eb.G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = eb.K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = eb.sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X] = parameters
    alphab = alpha_B(G3,G4,K, M_pG4=M_pG4,M_KG4=M_KG4,M_G3s=M_G3s,M_sG4=M_sG4,M_G3G4=M_G3G4,M_Ks=M_Ks,omegam=omegam)
    alphak = alpha_K(G3,G4,K, M_pG4=M_pG4,M_KG4=M_KG4,M_G3s=M_G3s,M_sG4=M_sG4,M_G3G4=M_G3G4,M_Ks=M_Ks,omegam=omegam)
    alphat = alpha_T(G3,G4,K, M_pG4=M_pG4,M_KG4=M_KG4,M_G3s=M_G3s,M_sG4=M_sG4,M_G3G4=M_G3G4,M_Ks=M_Ks,omegam=omegam)
    alpham = alpha_M(G3,G4,K, M_pG4=M_pG4,M_KG4=M_KG4,M_G3s=M_G3s,M_sG4=M_sG4,M_G3G4=M_G3G4,M_Ks=M_Ks,omegam=omegam)
    alphabprime = alpha_B_prime(G3,G4,K, M_pG4=M_pG4,M_KG4=M_KG4,M_G3s=M_G3s,M_sG4=M_sG4,M_G3G4=M_G3G4,M_Ks=M_Ks,omegam=omegam)
    D = alphak + (3./2.)*alphab*alphab
    rhom_tilde_by_Hsq = (3.*omegam*M_pG4*M_pG4) / (2.*(G4 -2.*X*G4x)  )
    numerator = (2.-alphab)*( Eprime/E - 0.5*alphab*(1+alphat) - (alpham - alphat)) - alphabprime + rhom_tilde_by_Hsq
    c_s_sq = -1.*numerator #/D
    return c_s_sq