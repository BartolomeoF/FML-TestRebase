import numpy as np
import matplotlib.pyplot as plt
import os

a_LCDM, E_LCDM, EprimeE_LCDM, Omega_m_LCDM, Omega_r_LCDM, Omega_l_LCDM = np.loadtxt('LCDM.txt', unpack=True)

filename ='WMAP_model_1_expansion_1.txt' #'{cosmology_name}_model_1_expansion.txt'

a_list, E_list, phi_list, phiprime_list, Omega_m_list, Omega_r_list, Omega_lambda_list, Omega_phi_list = [], [], [], [], [], [], [], []
n = 1
while os.path.exists(filename):
    a, E, phi, phiprime, Omega_m, Omega_r, Omega_lambda, Omega_phi = np.loadtxt(filename, unpack=True)
    a_list.append(a), E_list.append(E), phi_list.append(phi), phiprime_list.append(phiprime), Omega_m_list.append(Omega_m), Omega_r_list.append(Omega_r), Omega_lambda_list.append(Omega_lambda), Omega_phi_list.append(Omega_phi)
    filename = filename.replace('model_{}'.format(n), 'model_{}'.format(n+1))
    n+=1

#converting scale factor to redshift
z = 1/a_list[0] - 1

#a_LCDM = E_LCDM = EprimeE_LCDM = Omega_m_LCDM = Omega_r_LCDM = Omega_l_LCDM = 1

plt.figure(dpi=300)
for phi in phi_list:
    plt.plot(z, phi, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel(r'$\phi$')
plt.xscale('log')
plt.show()

plt.figure(dpi=300)
for phiprime in phiprime_list:
    plt.plot(z, phiprime, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel(r'$\phi^{\prime}$')
plt.xscale('log')
plt.show()

plt.figure(dpi=300)
for Omega_phi in Omega_phi_list:
    plt.plot(z, Omega_phi, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel(r'$\Omega_{\phi}$')
plt.xscale('log')
plt.show()

plt.figure(dpi=300)
for E in E_list:
    plt.plot(z, E/E_LCDM, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel(r'$E [E_{\Lambda CDM}]$')
plt.xscale('log')
plt.show()

plt.figure(dpi=300)
for Omega_m in Omega_m_list:
    plt.plot(z, Omega_m/Omega_m_LCDM, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel(r'$\Omega_m [\Omega_{m,\Lambda CDM}]$')
plt.xscale('log')
plt.show()

plt.figure(dpi=300)
for Omega_r in Omega_r_list: 
    plt.plot(z, Omega_r/Omega_r_LCDM, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel(r'$\Omega_r [\Omega_{r,\Lambda CDM}]$')
plt.xscale('log')
plt.show()

plt.figure(dpi=300)
for Omega_lambda in Omega_lambda_list:
    plt.plot(z, Omega_lambda/Omega_l_LCDM, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel(r'$\Omega_{\Lambda} [\Omega_{\Lambda,\Lambda CDM}]$')
plt.xscale('log')
plt.show()