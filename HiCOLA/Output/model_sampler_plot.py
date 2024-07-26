import numpy as np
import corner
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import ListedColormap
import matplotlib as mpl

def w0wa(a, w0, wa):
    """
    Functional form of w0wa parameterisation.
    """
    return w0+wa*(1-a)

def parameterise_w_phi(z_arr, z_max, w_phi_arr):
    """
    Finds best fit parameters for w0wa parameterisation of w_phi and evaluates goodness of fit using reduced chisquare.
    """
    a_arr = 1/(z_arr+1)
    a_fit = a_arr[z_arr<z_max]
    w_phi_data = w_phi_arr[z_arr<z_max]

    (w0, wa), junk = curve_fit(w0wa, a_fit, w_phi_data)

    w0wa_arr = w0wa(a_arr, w0, wa)[z_arr<z_max]

    model_params = (w0, wa)

    dof = len(a_fit)-2
    chisq = np.sum(((w_phi_data-w0wa_arr)**2.0)/(w0wa_arr**2.0))
    r_chisq = chisq/dof
    return w0wa_arr, model_params, r_chisq

#loading data
samples = np.loadtxt('samples_10_5.txt')
probabilities = np.loadtxt('probabilities_10_5.txt')
z, E_LCDM, best_fit_E, med_model, spread = np.loadtxt('posterior-input_10_5.txt', unpack=True)
best_models = np.loadtxt('best-models_10_5.txt')

#values for top 25 models
best_Es = best_models[:, 1:25]
best_w_phis = best_models[:, 25:]
theta_max = samples[np.argmax(probabilities)]

#plotting best models compared to LCDM
plt.figure(dpi=300)
plt.plot(z, E_LCDM, label=r'$\Lambda CDM$', color='k')
plt.plot(z, best_fit_E, label='Highest likelihood')
plt.plot(z, med_model, label='Median model')
plt.fill_between(z, med_model-spread, med_model+spread, color='grey', alpha=0.5, label=r'$1\sigma$ Posterior Spread')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('z')
plt.ylabel('E')
plt.xlim(right = 10)
plt.ylim(top = 10)
plt.show()

#----plotting ratio of best models to LCDM----
plt.figure(dpi=300)
plt.plot(z, best_fit_E/E_LCDM, label='Highest likelihood model')
plt.plot(z, med_model/E_LCDM, label='Median model')
plt.fill_between(z, (med_model-spread)/E_LCDM, (med_model+spread)/E_LCDM, color='grey', alpha=0.5, label=r'$1\sigma$ Posterior Spread')
#plt.plot(z, (best_Es.T/E_LCDM).T)
plt.legend()
plt.xscale('log')
plt.xlabel('z')
plt.ylabel(r'$H/H_{\Lambda CDM}$')
plt.show()

#----plotting w_phi----
z_max = 4.2
z_small = z[z<z_max]
DESICMB = -0.45-1.79*(z_small/(z_small+1))
DESICMBHi = -0.11-1.33*(z_small/(z_small+1))
DESICMBLo = -0.66-2.79*(z_small/(z_small+1))

DESICMBPantheonPlus = -0.827-0.75*(z_small/(z_small+1))
DESICMBPantheonPlusHi = -0.764-0.46*(z_small/(z_small+1)) 
DESICMBPantheonPlusLo = -0.89-1*(z_small/(z_small+1))

DESICMBUnion3 = -0.64-1.27*(z_small/(z_small+1))
DESICMBUnion3Hi = -0.53-0.87*(z_small/(z_small+1))
DESICMBUnion3Lo = -0.75-1.61*(z_small/(z_small+1))

DESICMBDESY5 = -0.727-1.05*(z_small/(z_small+1))
DESICMBDESY5Hi = -0.66-0.74*(z_small/(z_small+1))
DESICMBDESY5Lo = -0.794-1.32*(z_small/(z_small+1))

w_phi_fitted, params, r_chisq = parameterise_w_phi(z, z_max, best_w_phis[:,0])
print('Fitted parameters: w0 = {}, wa = {} \nReduced chi^2 = {}'.format(params[0], params[1], r_chisq))

plt.figure(dpi=300)
plt.fill_between(z_small, DESICMBLo, DESICMBHi, alpha=0.5, color='tab:purple')
plt.fill_between(z_small, DESICMBPantheonPlusLo, DESICMBPantheonPlusHi, alpha=0.5, color='tab:blue')
plt.fill_between(z_small, DESICMBUnion3Lo, DESICMBUnion3Hi, alpha=0.5, color='tab:orange')
plt.fill_between(z_small, DESICMBDESY5Lo, DESICMBDESY5Hi, alpha=0.5, color='tab:green')
plt.plot(z_small, w_phi_fitted, color='red')
plt.plot(z_small, best_w_phis[z<z_max], color='k', alpha=0.2, linewidth=1)
plt.xscale('log')
plt.xlabel('z')
plt.ylabel(r'$\omega_{\phi}$')
plt.legend(['DESI+CMB', 'DESI+CMB+PantheonPlus', 'DESI+CMB+Union3', 'DESI+CMB+DESY5', r'Best $\omega_0\omega_a$ fit', 'Top 25 models'])
plt.show()

#----corner plot----
levels = (0.118, 0.393, 0.675, 0.864, 0.96)
viridis = mpl.colormaps['viridis_r'].resampled(1000)
newcolors = viridis(np.linspace(0.2, 1, 1000))
transparent = np.array([1, 1, 1, 0])
newcolors[:9, :] = transparent
newcmp = ListedColormap(newcolors)

labels = [r'$k_{\phi}$', r'$k_X$', r'$g_{3\phi}$', r'$g_{3X}$', r'$g_{4\phi}$']
display_labels = [r'$k_{\phi}$', r'$k_X$', r'$g_{3X}$', r'$g_{4\phi}$']
ind = [0, 1, 3, 4]
corner.corner(samples[:, ind], show_titles=True, labels=display_labels, plot_datapoints=True, levels=levels, fill_contours=True, contourf_kwargs={'cmap':newcmp, 'colors':None})
plt.show()

#----plotting sample variation with step number----
steps = 900 #total number of iterations
fig, axes = plt.subplots(5, figsize=(8, 8), sharex=True)
labels = [r'$k_{\phi}$', r'$k_X$', r'$g_{3\phi}$', r'$g_{3X}$', r'$g_{4\phi}$']
for i in range(len(labels)):
    ax = axes[i]
    ax.plot(np.reshape(samples[:,i], (steps,-1)),color='k', alpha=0.1)
    ax.set_xlim(0, steps)
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel("step number");
plt.show()