import numpy as np
import matplotlib.pyplot as plt
import os

filename ='WMAP_model_1_expansion.txt' #'{cosmology_name}_model_1_expansion.txt'

a_list, E_list, phi_list, phiprime_list, Omega_m_list, Omega_r_list, Omega_lambda_list, Omega_phi_list = [], [], [], [], [], [], [], []
n = 1
while os.path.exists(filename):
    a, E, phi, phiprime, Omega_m, Omega_r, Omega_lambda, Omega_phi = np.loadtxt(filename, unpack=True)
    a_list.append(a), E_list.append(E), phi_list.append(phi), phiprime_list.append(phiprime), Omega_m_list.append(Omega_m), Omega_r_list.append(Omega_r), Omega_lambda_list.append(Omega_lambda), Omega_phi_list.append(Omega_phi)
    filename = filename.replace('model_{}'.format(n), 'model_{}'.format(n+1))
    n+=1

#converting scale factor to redshift
z = 1/a_list[0] - 1
#omega_phi_cl = 1 - (Omega_m + Omega_r + Omega_lambda)

plt.figure(dpi=300)
for phi in phi_list:
    plt.plot(z, phi, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel('phi')
plt.xscale('log')
plt.show()

plt.figure(dpi=300)
for phiprime in phiprime_list:
    plt.plot(z, phiprime, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel('phiprime')
plt.xscale('log')
plt.show()

plt.figure(dpi=300)
for E in E_list:
    plt.plot(z, E, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel('E')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.figure(dpi=300)
for Omega_m in Omega_m_list:
    plt.plot(z, Omega_m, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel('Omega_m')
plt.xscale('log')
plt.show()

plt.figure(dpi=300)
for Omega_r in Omega_r_list:    
    plt.plot(z, Omega_r, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel('Omega_r')
plt.xscale('log')
plt.show()

plt.figure(dpi=300)
for Omega_phi in Omega_phi_list:
    plt.plot(z, Omega_phi, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel('Omega_phi')
plt.xscale('log')
plt.show()

# plt.figure(dpi=300)
# plt.plot(z, omega_phi_cl, label=r'$f_{\phi} = 1.0$')
# #plt.legend(fontsize='large')
# plt.xlabel('z')
# plt.ylabel('Omega_phi_closure')
# plt.xscale('log')
# plt.show()

plt.figure(dpi=300)
for Omega_lambda in Omega_lambda_list:
    plt.plot(z, Omega_lambda, label=r'$f_{\phi} = 1.0$')
#plt.legend(fontsize='large')
plt.xlabel('z')
plt.ylabel('Omega_lambda')
plt.xscale('log')
plt.show()