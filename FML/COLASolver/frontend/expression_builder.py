import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sym
sym.init_printing()


#IMPORTANT: The dimensionless field, called \Tilde{\phi} in notes, is called
# "phi" in this script, just like the original massive scalar field. Only the 
# SymPy definition carries any indication that it is really the dimensionless field!

#Any new mass scales in other terms like the G-functions should be specified by the user
# when defining them. That is, rather than hard-coding M_s^2 wherever there is an X,
# make sure to define X = (M_s^2/2)(\partial_t \phi)^2, etc.

#---Do not change, this remains the same for any model----
a, E, Eprime, phi, phiprime, phiprimeprime, X,  M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, M_gp, omegar, omegam, omegal, f, Theta = sym.symbols("a E Eprime phi phiprime phiprimeprime X M_{pG4} M_{KG4} M_{G3s} M_{sG4} M_{G3G4} M_{Ks} M_{gp} Omega_r Omega_m Omega_l f Theta")
#------------------------

def K_func(K,subscript='default',printswitch=0):
    if subscript=='X':
        subscript = sym.Symbol(subscript)
        Kx = sym.diff(K,X)
        return Kx
    if subscript=='XX':
        subscript = sym.Symbol(subscript)
        Kx = sym.diff(K,X)
        Kxx = sym.diff(Kx,X)
        return Kxx
    if subscript=='X,phi':
        subscript = sym.Symbol(subscript)
        Kx = sym.diff(K,X)
        Kxphi = sym.diff(Kx,phi)
        return Kxphi
    if subscript=='phi':
        Kphi = sym.diff(K,phi)
        return Kphi
    else:
        Kx = sym.diff(K,X)
        Kxx = sym.diff(Kx,X)
        Kxphi = sym.diff(Kx,phi)
        Kphi = sym.diff(K,phi)
        if printswitch==1:
            print('Input: K='+str(K))
            print('Output 1: Kx='+str(Kx))
            print('Output 2: Kxx='+str(Kxx))
            print('Output 3: Kxphi='+str(Kxphi))
            print('Output 4: Kphi='+str(Kphi))
        return Kx, Kxx, Kxphi, Kphi

def G3_func(G3,subscript='default',printswitch=0):
    if subscript=='X':
        subscript = sym.Symbol(subscript)
        G3x = sym.diff(G3,X)
        return G3x
    if subscript=='XX':
        subscript = sym.Symbol(subscript)
        G3x = sym.diff(G3,X)
        G3xx = sym.diff(G3x,X)
        return G3xx
    if subscript=='X,phi':
        subscript = sym.Symbol(subscript)
        G3x = sym.diff(G3,X)
        G3xphi = sym.diff(G3x,phi)
        return G3xphi
    if subscript=='phi,X':
        G3phi = sym.diff(G3,phi)
        G3phix = sym.diff(G3phi,X)
        return G3phix
    if subscript=='phi,phi':
        subscript = sym.Symbol(subscript)
        G3phi = sym.diff(G3,phi)
        G3phiphi = sym.diff(G3phi,phi)
        return G3phiphi
    if subscript=='phi':
        subscript=sym.Symbol(subscript)
        G3phi = sym.Symbol(G3,phi)
        return G3phi
    else:
        G3x = sym.diff(G3,X)
        G3phi = sym.diff(G3,phi)
        G3phiphi = sym.diff(G3phi,phi)
        G3xx = sym.diff(G3x,X)
        G3xphi = sym.diff(G3x,phi)
        G3phix = sym.diff(G3phi,X)
        if printswitch==1:
            print('Input: G3='+str(G3))
            print('Output 1: G3x='+str(G3x))
            print('Output 2: G3xx='+str(G3xx))
            print('Output 3: G3xphi='+str(G3xphi))
            print('Output 4: G3phix='+str(G3phix))
            print('Output 5: G3phiphi='+str(G3phiphi))
            print('Output 6: G3phi ='+str(G3phi))
        return G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi

def G4_func(G4,subscript='default',printswitch=0):
    if subscript=='X':
        subscript = sym.Symbol(subscript)
        G4x = sym.diff(G4,X)
        return G4x
    if subscript=='XX':
        subscript = sym.Symbol(subscript)
        G4x = sym.diff(G4,X)
        G4xx = sym.diff(G4x,X)
        return G4xx
    if subscript=='X,phi':
        subscript = sym.Symbol(subscript)
        G4x = sym.diff(G4,X)
        G4xphi = sym.diff(G4x,phi)
        return G4xphi
    if subscript=='phi,X':
        G4phi = sym.diff(G4,phi)
        G4phix = sym.diff(G4phi,X)
        return G4phix
    if subscript=='phi,phi':
        subscript = sym.Symbol(subscript)
        G4phi = sym.diff(G4,phi)
        G4phiphi = sym.diff(G4phi,phi)
        return G4phiphi
    else:
        G4x = sym.diff(G4,X)
        G4xx = sym.diff(G4x,X)
        G4xphi = sym.diff(G4x,phi)
        G4phi = sym.diff(G4,phi)
        G4phix = sym.diff(G4phi,X)
        G4phiphi = sym.diff(G4phi,phi)
        if printswitch==1:
            print('Input: G4='+str(G4))
            print('Output 1: G4x='+str(G4x))
            print('Output 2: G4xx='+str(G4xx))
            print('Output 3: G4xphi='+str(G4xphi))
            print('Output 4: G4phix='+str(G4phix))
            print('Output 5: G4phiphi'+str(G4phiphi))
            print('Output 6: G4phi='+str(G4phi))
        return G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi

def sy(string):
    return sym.Symbol(string)




def Pde(G3, G4,  K, 
        H='H',
        Hprime = 'Hprime',
        Meff='M_eff',
        Mp='M_p',
        Ms='M_s',
        phi='Tildephi',
        phiprime='Tildephiprime',
        phiprimeprime='Tildephiprimeprime',
        omegar='Omega_r',
        X='X'):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    parameters = [G3, G4, H, Hprime, K, Meff, Mp, phi, phiprime, phiprimeprime, omegar,X, Ms]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str):
                parameters[i] = sy(parameters[i])
    H = parameters[2] #assuming non-default arguments will already be sympy symbols
    Hprime = parameters[3]
    K = parameters[4]
    Meff = parameters[5]
    Mp = parameters[6]
    phi = parameters[7]
    phiprime = parameters[8]
    phiprimeprime = parameters[9]
    # phidot = phiprime*H
    # phidotdot = H*(Hprime*phiprime + H*phiprimeprime)
    omegar = parameters[9]
    X = parameters[10]
    Ms = parameters[-1]
    Mfrac = (Mp**2)/(Meff**2)
    term1 = 2*X* (G3phi + H*(Hprime*Ms*phiprime + H*Ms*phiprimeprime) *G3x)
    term2 = 2* G4phi* (H*(Hprime*Ms*phiprime + H*Ms*phiprimeprime) + 2 *(H**2) *Ms*phiprime) 
    term3 = 4 *X *G4phiphi
    P_DE = Mfrac*(K-term1 + term2 + term3) + ((H**2)*omegar*(Mp**2))* (Mfrac - 1)
    return P_DE

def rhode(G3, G4,  K,
        H='H',
        Meff='M_eff',
        Mp='M_p',
        Ms='M_s',
        phi='phi',
        phidot='phidot',
        phidotdot='phidotdot',
        omegam='Omega_m',
        omegar='Omega_r',
        X='X'): #strings as inputs
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)    
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    parameters = [G3, G4, H, K, Meff, Mp, phi, phidot, phidotdot, omegar,X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str):
                parameters[i] = sy(parameters[i])
    Mfrac = (Mp**2)/(Meff**2)
    term1 = 2* X* Kx - K + 6 *X* Ms*phiprime* (H**2)*G3x
    term2 = 2* X* G3phi + 6* (H**2) *Ms*phiprime* G4phi
    return (3*(H**2)*(Mp**2))(omegar + omegam) *( Mfrac - 1) + Mfrac* (term1 - term2)

def omega_phi(G3, G4,  K, 
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
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, X] = parameters
    term1 = (omegam + omegar + omegal)*( (M_pG4**2.)/(2.*G4) - 1. )
    term21 = (M_KG4**2.)*X*Kx/(E**2.) - (M_KG4**2.)*K/(2.*(E**2.))
    term22 = 3*M_G3s*M_sG4*X*phiprime*G3x - M_G3G4*M_sG4*X*G3phi/(E**2.) - 3*phiprime*G4phi
    term2 = term21 + term22
    omega_de = term1 + (1/(3.*G4))*term2
    return omega_de


def EprimeEODERHS(G3, G4,  K, 
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
        omegal='Omega_l',
        X='X'):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegal, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegal, X] = parameters
    M_G4s = 1./M_sG4
    A1 = (M_Ks**2.)*Kx - M_G3s*G3phi + 2.*X*M_G3s*G3phix
    A2 = 6*(E**2)*phiprime*(M_G3s*G3x + X*M_G3s*G3xx) + (E**2.)*(phiprime**2.)*( (M_Ks**2.)*Kxx - 2.*M_G3s*G3phix)
    A = A1 + A2
    # print('A is')
    # print(sym.latex(A))
    B1 = 6.*M_G3s*X*G3x - 6.*(M_G4s**2)*G4phi
    # print('B1 is')
    # print(sym.latex(B1))
    B21 = 3.*phiprime*((M_Ks**2.)*Kx - 2.*M_G3s*G3phi + 2.*M_G3s*X*G3phix)
    B22 = (phiprime**2.)*( (M_Ks**2.)*Kxphi - 2.*M_G3s*G3phiphi) - ((M_Ks**2.)/(E**2.))*Kphi
    B23 = -12.*(M_G4s**2.)*G4phi + 18.*M_G3s*X*G3x + 2.*M_G3s*X*G3phiphi*(1./E**2.)
    B2 = B21 + B22 + B23
    # print('B2 is')
    # print(sym.latex(B2))
    term1 = 1.+ (1./2.)*M_G3G4*M_sG4*X*G3x*(1./G4)*(B1/A) - (G4phi*B1)/(2*G4*A)
    # print('term1 is')
    # print(sym.latex(term1))
    term21 = (M_KG4**2.)*K*(1./(E**2.)) - 2.*M_sG4*M_G3G4*(1./(E**2.))*X*G3phi
    term22 = 4.*G4phi*phiprime + 4.*X*G4phiphi*(1./(E**2.))
    term2 = (-1./(4.*G4))*(term21 + term22)
    # print('term2 is')
    # print(sym.latex(term2))
    term3 = (-1./2.)*( ( (omegar - 3.*omegal )/(2.*G4))*(M_pG4**2.)  + 3. ) #cosmological constant has been added!
    term4 = (-1./2.)*M_G3G4*M_sG4*X*G3x*(B2/(G4*A)) + (G4phi*B2)/(2*G4*A)
    RHS = term2 + term3 + term4
    # print('RHS is')
    # print(sym.latex(RHS))
    EprimeE =  (term2 + term3 + term4)/term1 
    return EprimeE

def EprimeEODERHS_safe(G3, G4,  K,    
        threshold='threshold', 
        threshold_sign='threshold_sign',
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
        omegal ='Omega_l',
        X='X'):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegal, X, threshold, threshold_sign]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegal, X, threshold, threshold_sign] = parameters
    M_G4s = 1./M_sG4
    A = threshold*threshold_sign
    # print('A is')
    # print(sym.latex(A))
    B1 = 6.*M_G3s*X*G3x - 6.*(M_G4s**2)*G4phi
    # print('B1 is')
    # print(sym.latex(B1))
    B21 = 3.*phiprime*((M_Ks**2.)*Kx - 2.*M_G3s*G3phi + 2.*M_G3s*X*G3phix)
    B22 = (phiprime**2.)*( (M_Ks**2.)*Kxphi - 2.*M_G3s*G3phiphi) - ((M_Ks**2.)/(E**2.))*Kphi
    B23 = -12.*(M_G4s**2.)*G4phi + 18.*M_G3s*X*G3x + 2.*M_G3s*X*G3phiphi*(1./E**2.)
    B2 = B21 + B22 + B23
    # print('B2 is')
    # print(sym.latex(B2))
    term1 = 1.+ (1./2.)*M_G3G4*M_sG4*X*G3x*(1./G4)*(B1/A) - (G4phi*B1)/(2*G4*A)
    # print('term1 is')
    # print(sym.latex(term1))
    term21 = (M_KG4**2.)*K*(1./(E**2.)) - 2.*M_sG4*M_G3G4*(1./(E**2.))*X*G3phi
    term22 = 4.*G4phi*phiprime + 4.*X*G4phiphi*(1./(E**2.))
    term2 = (-1./(4.*G4))*(term21 + term22)
    # print('term2 is')
    # print(sym.latex(term2))
    term3 = (-1./2.)*( (omegar/(2.*G4))*(M_pG4**2.) - 3.*omegal + 3. )
    term4 = (-1./2.)*M_G3G4*M_sG4*X*G3x*(B2/(G4*A)) + (G4phi*B2)/(2*G4*A)
    RHS = term2 + term3 + term4
    # print('RHS is')
    # print(sym.latex(RHS))
    EprimeE =  (term2 + term3 + term4)/term1 
    return EprimeE

def phiprimeprimeODERHS(G3, G4,  K, 
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
        X='X'):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    M_G4s = 1./M_sG4
    A1 = (M_Ks**2.)*Kx - M_G3s*G3phi + 2.*X*M_G3s*G3phix
    A2 = 6*(E**2)*phiprime*(M_G3s*G3x + X*M_G3s*G3xx) + (E**2.)*(phiprime**2.)*( (M_Ks**2.)*Kxx - 2.*M_G3s*G3phix)
    A = A1 + A2
    B1 = 6.*M_G3s*X*G3x - 6.*(M_G4s**2)*G4phi
    B21 = 3.*phiprime*((M_Ks**2.)*Kx - 2.*M_G3s*G3phi + 2.*M_G3s*X*G3phix)
    B22 = (phiprime**2.)*( (M_Ks**2.)*Kxphi - 2.*M_G3s*G3phiphi) - ((M_Ks**2.)/(E**2.))*Kphi
    B23 = -12.*(M_G4s**2.)*G4phi + 18.*M_G3s*X*G3x + 2.*M_G3s*X*G3phiphi*(1./E**2.)
    B2 = B21 + B22 + B23
    B =  B1*(Eprime/E)+B2
    phiprimeprime = -1.*(  (B/A)  + (Eprime/E)*phiprime )
    return phiprimeprime

#this function is obtained by substituting eq. 90 (omegade function) in Ashim's overleaf into eq. 8
def fried_closure(G3, G4,  K,
        f='f',
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
        omegal='Omega_l',
        X='X'):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    param = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, f, X]
    paramnum = len(param)
    for i in np.arange(0,paramnum):
            if isinstance(param[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                param[i] = sy(param[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, omegal, f, X] = param
    omega_field = omega_phi(G3,G4,K,M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4, M_Ks=M_Ks)
    fried1_RHS = omega_field + omegam + omegar + omegal -1.
    Xreal = (1./2.)*(E**2.)*(phiprime**2.)
    fried1_RHS = fried1_RHS.subs(X,Xreal)
    # print('fried1 SymPy is ')
    # print(sym.latex(fried1_RHS))
    # if (model=='cubic' or model=='cubic Galileon' or model=='cubic_Galileon' or model=='Cubic Galileon'):
    #     fried1_RHS_lambda = sym.lambdify([phiprime,E,omegar,omegam,k1,g31],fried1_RHS)
    # elif model=='Traykova':
    #     fried1_RHS_lambda = sym.lambdify([phiprime,E,omegar,omegam,k1,k2, g31, g32],fried1_RHS)
    return fried1_RHS

def phiprimeprimeODERHS_safe(G3, G4,  K,      
        threshold='threshold', 
        threshold_sign='threshold_sign',
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
        X='X'):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X, threshold, threshold_sign]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X, threshold, threshold_sign] = parameters
    M_G4s = 1./M_sG4
    A = threshold*threshold_sign
    B1 = 6.*M_G3s*X*G3x - 6.*(M_G4s**2)*G4phi
    B21 = 3.*phiprime*((M_Ks**2.)*Kx - 2.*M_G3s*G3phi + 2.*M_G3s*X*G3phix)
    B22 = (phiprime**2.)*( (M_Ks**2.)*Kxphi - 2.*M_G3s*G3phiphi) - ((M_Ks**2.)/(E**2.))*Kphi
    B23 = -12.*(M_G4s**2.)*G4phi + 18.*M_G3s*X*G3x + 2.*M_G3s*X*G3phiphi*(1./E**2.)
    B2 = B21 + B22 + B23
    B =  B1*(Eprime/E)+B2
    phiprimeprime = -1.*(  (B/A)  + (Eprime/E)*phiprime )
    return phiprimeprime

def A_func(G3, G4,  K, 
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
        X='X'):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    M_G4s = 1./M_sG4
    A1 = (M_Ks**2.)*Kx - M_G3s*G3phi + 2.*X*M_G3s*G3phix
    A2 = 6*(E**2)*phiprime*(M_G3s*G3x + X*M_G3s*G3xx) + (E**2.)*(phiprime**2.)*( (M_Ks**2.)*Kxx - 2.*M_G3s*G3phix)
    A = A1 + A2
    return A

def B2_func(G3, G4,  K,    
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
        X='X'):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    M_G4s = 1./M_sG4
    B1 = 6.*M_G3s*X*G3x - 6.*(M_G4s**2)*G4phi
    # print('B1 is')
    # print(sym.latex(B1))
    B21 = 3.*phiprime*((M_Ks**2.)*Kx - 2.*M_G3s*G3phi + 2.*M_G3s*X*G3phix)
    B22 = (phiprime**2.)*( (M_Ks**2.)*Kxphi - 2.*M_G3s*G3phiphi) - ((M_Ks**2.)/(E**2.))*Kphi
    B23 = -12.*(M_G4s**2.)*G4phi + 18.*M_G3s*X*G3x + 2.*M_G3s*X*G3phiphi*(1./E**2.)
    B2 = B21 + B22 + B23
    return B2


def theta(G3, G4,  K, 
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
        X='X',
        print_flag=0,
        simplify_flag=0):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    term1 = M_sG4*M_G3G4*E*phiprime*X*G3x/M_pG4/M_pG4
    term2 = 2.*E*G4/M_pG4/M_pG4
    term3 = E*phiprime*G4phi/M_pG4/M_pG4
    theta = -1.*term1 + term2 + term3
    Xreal = 0.5*(E**2.)*phiprime**2.
    theta = theta.subs(X, Xreal)
    if print_flag == 1:
        if simplify_flag == 0:
            print('---- THETA --------')
            print(sym.latex(theta))
            print('-------------------')
        elif simplify_flag == 1:
            theta_simple = sym.simplify(theta)
            print('---- SIMPLIFIED THETA --------')
            print(sym.latex(theta_simple))
            print('-------------------')
    return theta

def calE(G3, G4,  K, 
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
        X='X',
        print_flag=0,
        simplify_flag=0):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    term1 = 2.*M_KG4*M_KG4*X*Kx/M_pG4/M_pG4
    term2 = M_KG4*M_KG4*K/M_pG4/M_pG4
    term3 = 6.*M_sG4*M_G3G4*E*E*X*G3x*phiprime/M_pG4/M_pG4
    term4 = 2.*M_sG4*M_G3G4*X*G3phi/M_pG4/M_pG4
    term5 = 6.*E*E*G4/M_pG4/M_pG4
    term6 = 6.*E*E*G4phi*phiprime/M_pG4/M_pG4
    calE = term1 - term2 + term3 - term4 - term5 - term6
    if print_flag == 1:
        if simplify_flag == 0:
            print('---- calE --------')
            print(sym.latex(calE))
            print('-------------------')
        elif simplify_flag == 1:
            calE_simple = sym.simplify(calE)
            print('---- SIMPLIFIED calE --------')
            print(sym.latex(calE_simple))
            print('-------------------')
    return calE

def calP(G3, G4,  K, 
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
        X='X',
        print_flag =0,
        simplify_flag=0):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    term1 = M_KG4*M_KG4*K/M_pG4/M_pG4
    term2coeff = 2.*M_sG4*X/M_pG4
    term2bracket = M_G3s*G3phi + E*M_G3s*G3x*(Eprime*phiprime + E*phiprimeprime)
    term2 = term2coeff*term2bracket
    term3 = 2.*G4*(3.*E*E + 2*E*Eprime)/M_pG4/M_pG4
    term4coeff = 2*G4phi/M_pG4/M_pG4
    term4bracket = E*(Eprime*phiprime + E*phiprimeprime) + 2*E*E*phiprime
    term4 = term4coeff*term4bracket
    calP = term1 - term2 + term3 + term4
    if print_flag == 1:
        if simplify_flag == 0:
            print('---- calP --------')
            print(sym.latex(calP))
            print('-------------------')
        elif simplify_flag == 1:
            calP_simple = sym.simplify(calP)
            print('---- SIMPLIFIED calP --------')
            print(sym.latex(calP_simple))
            print('-------------------')
    return calP

def alpha0(G3, G4,  K, 
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
        X='X',
        Thetaprime='Thetaprime',
        G4prime='G4prime',
        print_flag =0,
        simplify_flag=0): #Thetaprime to be supplied numerically, possibly after interpolation
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    calliE = calE(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    calliP = calP(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    Theta = theta(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    # Xreal = 0.5*(E**2.)*phiprime**2.
    # Theta = Theta.subs(X,Xreal)
    dTheta_dE = sym.diff(Theta,E)#-1.*M_sG4*M_G3G4*phiprime*X*G3x/M_pG4/M_pG4 - M_sG4*M_G3G4*E*E*phiprime*phiprime*phiprime*G3x/M_pG4/M_pG4 + 2.*G4/M_pG4/M_pG4 + phiprime*G4phi/M_pG4/M_pG4
    # dTheta_dG3x = sym.diff(Theta,G3x)
    # dTheta_dG4 = sym.diff(Theta,G4)
    # dTheta_dG4phi = sym.diff(Theta,G4phi)
    dTheta_dphiprime = sym.diff(Theta,phiprime)#-1.*M_sG4*M_G3G4*(X*G3x + phiprime*phiprime*E*E*G3x)/M_pG4/M_pG4 + E*G4phi/M_pG4/M_pG4 #This was added in meeting on 2Aug; to remove X so that total derivative makes sense
    Thetaprime = dTheta_dE*Eprime + dTheta_dphiprime*phiprimeprime
    # Thetaprime2 = dTheta_dG3x*(G3xx*(E*Eprime*phiprime*phiprime + E*E*phiprime*phiprimeprime) + G3xphi*phiprime  )
    # Thetaprime3 = dTheta_dG4*G4phi*phiprime + dTheta_dG4phi*G4phiphi*phiprime
    # Thetaprime = Thetaprime1 + Thetaprime2 + Thetaprime3
    A0 = Thetaprime/E + Theta/E - 2.*G4/M_pG4/M_pG4 - 4.*G4phi*phiprime/M_pG4/M_pG4 - (calliE + calliP)/(2.*E*E)
    alpha0 = M_pG4*M_pG4*A0/2./G4
    if print_flag == 1:
        if simplify_flag == 0:
            print('---- alpha0 --------')
            print(sym.latex(alpha0))
            print('-------------------')
        elif simplify_flag == 1:
            alpha0_simple = sym.simplify(alpha0)
            print('---- SIMPLIFIED alpha0 --------')
            print(sym.latex(alpha0_simple))
            print('-------------------')
    return alpha0

def alpha1(G3, G4,  K, 
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
        X='X',
        G4prime='G4prime',
        print_flag =0,
        simplify_flag=0): #Thetaprime to be supplied numerically, possibly after interpolation
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    A1 = 2.*G4phi*phiprime/M_pG4/M_pG4
    alpha1 = M_pG4*M_pG4*A1/2./G4
    if print_flag == 1:
        if simplify_flag == 0:
            print('---- alpha1 --------')
            print(sym.latex(alpha1))
            print('-------------------')
        elif simplify_flag == 1:
            alpha1_simple = sym.simplify(alpha1)
            print('---- SIMPLIFIED alpha1 --------')
            print(sym.latex(alpha1_simple))
            print('-------------------')
    return alpha1


def alpha2(G3, G4,  K, 
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
        X='X',
        Thetaprime='Thetaprime',
        G4prime='G4prime',
        print_flag =0,
        simplify_flag=0): #Thetaprime to be supplied numerically, possibly after interpolation
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    Theta = theta(G3,G4, K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    A2 = 2.*G4/M_pG4/M_pG4 - Theta/E
    alpha2 = M_pG4*M_pG4*A2/2./G4
    if print_flag == 1:
        if simplify_flag == 0:
            print('---- alpha2 --------')
            print(sym.latex(alpha2))
            print('-------------------')
        elif simplify_flag == 1:
            alpha2_simple = sym.simplify(alpha2)
            print('---- SIMPLIFIED alpha2 --------')
            print(sym.latex(alpha2_simple))
            print('-------------------')
    return alpha2

def beta0(G3, G4,  K, 
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
        X='X',
        print_flag =0,
        simplify_flag=0):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    B0 = M_sG4*M_G3G4*X*G3x*phiprime/M_pG4/M_pG4
    beta0 = M_pG4*M_pG4*B0/2./G4
    if print_flag == 1:
        if simplify_flag == 0:
            print('---- beta0 --------')
            print(sym.latex(beta0))
            print('-------------------')
        elif simplify_flag == 1:
            beta0_simple = sym.simplify(beta0)
            print('---- SIMPLIFIED beta0 --------')
            print(sym.latex(beta0_simple))
            print('-------------------')
    return beta0

def calB(G3, G4,  K, 
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
        X='X',
        print_flag =0,
        simplify_flag=0):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    alpha_0 = alpha0(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    alpha_1 = alpha1(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    alpha_2 = alpha2(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    beta_0 = beta0(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    calB = 4.*beta_0/(alpha_0 + 2.*alpha_1*alpha_2 + alpha_2*alpha_2)
    if print_flag == 1:
        if simplify_flag == 0:
            print('---- calB --------')
            print(sym.latex(calB))
            print('-------------------')
        elif simplify_flag == 1:
            calB_simple = sym.simplify(calB)
            print('---- SIMPLIFIED calB --------')
            print(sym.latex(calB_simple))
            print('-------------------')
    return calB #this will depend on E, Eprime, phiprime, phiprimeprime, Theta, Thetaprime, G4prime

def calC(G3, G4,  K, 
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
        X='X',
        print_flag =0,
        simplify_flag=0):
    G3x, G3xx, G3xphi, G3phix, G3phiphi, G3phi = G3_func(G3)
    G4x, G4xx, G4xphi, G4phix, G4phiphi, G4phi = G4_func(G4)
    Kx, Kxx, Kxphi, Kphi = K_func(K)
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, X] = parameters
    alpha_0 = alpha0(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    alpha_1 = alpha1(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    alpha_2 = alpha2(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    calC = (alpha_1 + alpha_2)/(alpha_0 + 2.*alpha_1*alpha_2 + alpha_2*alpha_2)
    if print_flag == 1:
        if simplify_flag == 0:
            print('---- calC --------')
            print(sym.latex(calC))
            print('-------------------')
        elif simplify_flag == 1:
            calC_simple = sym.simplify(calC)
            print('---- SIMPLIFIED calC --------')
            print(sym.latex(calC_simple))
            print('-------------------')
    return calC #this will depend on E, Eprime, phiprime, phiprimeprime, Theta, Thetaprime, G4prime


def coupling_factor(G3, G4,  K, 
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
        omegam = 'Omega_m',
        X='X',
        a='a',
        print_flag=0,
        simplify_flag=0):
    parameters = [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, X, a]
    paramnum = len(parameters)
    for i in np.arange(0,paramnum):
            if isinstance(parameters[i],str): #these sympy variables are prob not globally defined, so need to always make sure there are corresponding global variables with the smae names for .subs to work?
                parameters[i] = sy(parameters[i])
    [G3, G4, K, E, Eprime, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, phi, phiprime, phiprimeprime, omegar, omegam, X, a] = parameters
    alpha_1 = alpha1(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    alpha_2 = alpha2(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks,print_flag=print_flag,simplify_flag=simplify_flag)
    C = calC(G3,G4,K, M_pG4=M_pG4, M_KG4=M_KG4, M_G3s=M_G3s, M_sG4=M_sG4, M_G3G4=M_G3G4,M_Ks=M_Ks)
    coupling = -1.*(alpha_1 + alpha_2)*C
    if print_flag == 1:
        if simplify_flag == 0:
            print('---- coupling factor --------')
            print(sym.latex(coupling))
            print('-------------------')
        elif simplify_flag == 1:
            coupling_simple = sym.simplify(coupling)
            print('---- SIMPLIFIED coupling factor --------')
            print(sym.latex(coupling_simple))
            print('-------------------')
    return coupling

def create_Horndeski(K,G3,G4,symbol_list,mass_ratio_list):
    if np.any([K,G3,G4]) is None:
        raise Exception("Horndeski functions K, G3 and G4 have not been specified.")
    
    [M_pG4_test, M_KG4_test, M_G3s_test, M_sG4_test, M_G3G4_test, M_Ks_test, M_gp_test] = mass_ratio_list
    E_prime_E = EprimeEODERHS(G3, G4, K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test) #These are the actual equations that need to use SymPy builder script
    Xreal = 0.5*(E**2.)*phiprime**2.
    
    E_prime_E = E_prime_E.subs(X,Xreal)
    E_prime_E_safe = EprimeEODERHS_safe(G3, G4, K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    E_prime_E_safe = E_prime_E_safe.subs(X,Xreal)
    
    
    phi_primeprime = phiprimeprimeODERHS(G3, G4, K,M_pG4=M_pG4_test, M_KG4 =M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    phi_primeprime = phi_primeprime.subs(X,Xreal)
    phi_primeprime_safe = phiprimeprimeODERHS_safe(G3, G4, K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    phi_primeprime_safe = phi_primeprime_safe.subs(X,Xreal)

    A_function = A_func(G3=G3, G4=G4, K=K,M_pG4=M_pG4_test, M_KG4 =M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    A_function = A_function.subs(X,Xreal)
    
    B2_function = B2_func(G3=G3, G4=G4, K=K,M_pG4=M_pG4_test, M_KG4 =M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    B2_function = B2_function.subs(X,Xreal)
    B2_lambda = sym.lambdify([E,phiprime,*symbol_list],B2_function,"scipy")
    
    omega_field = omega_phi(G3=G3,G4=G4,K=K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test, M_G3G4=M_G3G4_test, M_Ks=M_Ks_test)
    omega_field = omega_field.subs(X,Xreal)
    
    fried_RHS_lambda = fried_closure(G3, G4,  K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test, M_G3G4=M_G3G4_test, M_Ks=M_Ks_test)
    fried_RHS_lambda = fried_RHS_lambda.subs(X,Xreal)
    fried_RHS_lambda = sym.lambdify([E, phiprime, omegar,omegam, omegal, *symbol_list],fried_RHS_lambda)
    
    E_prime_E_lambda = sym.lambdify([E,phiprime,omegar,omegal,*symbol_list],E_prime_E, "scipy")
    E_prime_E_safelambda = sym.lambdify([E,phiprime,omegar, omegal, threshold,threshold_sign,*symbol_list],E_prime_E_safe, "scipy")
    
    phi_primeprime_lambda = sym.lambdify([E,Eprime,phiprime,*symbol_list],phi_primeprime, "scipy")
    phi_primeprime_safelambda = sym.lambdify([E,Eprime,phiprime,threshold,threshold_sign,*symbol_list],phi_primeprime_safe, "scipy")
    omega_phi_lambda = sym.lambdify([E,phiprime,*symbol_list],omega_field)
    A_lambda = sym.lambdify([E,phiprime,*symbol_list],A_function,"scipy")
    
    
    alpha0_func = alpha0(G3,G4,K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    alpha0_func = alpha0_func.subs(X,Xreal)
    alpha0_lamb = sym.lambdify([E,Eprime,phiprime,phiprimeprime, *symbol_list],alpha0_func)
    
    alpha1_func = alpha1(G3,G4,K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    alpha1_func = alpha1_func.subs(X,Xreal)
    alpha1_lamb = sym.lambdify([E,Eprime,phiprime,phiprimeprime, *symbol_list],alpha1_func)
    
    alpha2_func = alpha2(G3,G4,K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    alpha2_func = alpha2_func.subs(X,Xreal)
    alpha2_lamb = sym.lambdify([E,Eprime,phiprime,phiprimeprime, *symbol_list],alpha2_func)
    
    beta0_func = beta0(G3,G4,K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    beta0_lamb = sym.lambdify([E,Eprime,phiprime,phiprimeprime, *symbol_list],beta0_func)
    
    calB_func = calB(G3,G4,K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    calB_func = calB_func.subs(X,Xreal)
    calB_lamb = sym.lambdify([E,Eprime,phiprime,phiprimeprime, *symbol_list],calB_func)
    
    calC_func = calC(G3,G4,K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    calC_func = calC_func.subs(X,Xreal)
    calC_lamb = sym.lambdify([E,Eprime,phiprime,phiprimeprime, *symbol_list],calC_func)
    
    
    coupling_fac = coupling_factor(G3,G4,K, M_pG4=M_pG4_test, M_KG4=M_KG4_test, M_G3s=M_G3s_test, M_sG4=M_sG4_test,M_G3G4=M_G3G4_test,M_Ks=M_Ks_test)
    coupling_fac = coupling_fac.subs([(X,Xreal)])
    coupling_fac = sym.lambdify([E,Eprime,phiprime,phiprimeprime, *symbol_list],coupling_fac)
    return E_prime_E_lambda, E_prime_E_safelambda, phi_primeprime_lambda, phi_primeprime_safelambda, omega_phi_lambda, fried_RHS_lambda, A_lambda, B2_lambda, \
    coupling_fac, alpha0_lamb, alpha1_lamb, alpha2_lamb, beta0_lamb, calB_lamb, calC_lamb


def write_data_screencoupl(a_arr_inv, chioverdelta_arr, Coupl_arr, output_filename_as_string): #from Bill's Galileon coupling script
    datafile_id = open(output_filename_as_string, 'wb')    #here you open the ascii file
    data = np.array([np.array(a_arr_inv[::-1]), np.array(chioverdelta_arr[::-1]), np.array(Coupl_arr[::-1])])
    data = data.T     #here you transpose your data, so to have it in two columns
    np.savetxt(datafile_id, data, fmt=['%.4e','%.4e','%.4e'])    #here the ascii file is populated.
    datafile_id.close()    #close the file


E, Eprime, phi, phiprime, phiprimeprime, X, k1, g31, g32, k2, g4, M_pG4, M_KG4, M_G3s, M_sG4, M_G3G4, M_Ks, M_gp, omegar, omegam, omegam0, Theta, Thetaprime, G4prime, a, threshold, threshold_sign = sym.symbols("E Eprime phi phiprime phiprimeprime X k_1 g_{31} g_{32} k_2 g_4 M_{pG4} M_{KG4} M_{G3s} M_{sG4} M_{G3G4} M_{Ks} M_{gp} Omega_r Omega_m Omega_{m0} Theta Theta_prime G4prime a threshold threshold_sign")
