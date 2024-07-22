import numpy as np
import corner
import matplotlib.pyplot as plt

#loading data
samples = np.loadtxt('samples_10_5.txt')
probabilities = np.loadtxt('probabilities_10_5.txt')
z, E_LCDM, best_fit_model, med_model, spread = np.loadtxt('posterior-input_10_5.txt', unpack=True)

theta_max = samples[np.argmax(probabilities)]

#plotting best models compared to LCDM
plt.plot(z, E_LCDM, label='LCDM', color='k')
plt.plot(z, best_fit_model, label='Highest likelihood')
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

#plotting ratio of best models to LCDM
plt.plot(z, best_fit_model/E_LCDM, label='Highest likelihood model')
plt.plot(z, med_model/E_LCDM, label='Median model')
plt.fill_between(z, (med_model-spread)/E_LCDM, (med_model+spread)/E_LCDM, color='grey', alpha=0.5, label=r'$1\sigma$ Posterior Spread')
plt.legend()
plt.legend()
plt.xscale('log')
plt.xlabel('z')
plt.ylabel(r'$E/E_{\Lambda CDM}$')
plt.show()

#corner plot
labels = [r'$k_{\phi}$', r'$k_X$', r'$g_{3\phi}$', r'$g_{3X}$', r'$g_{4\phi}$']
corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16,0.5,0.84])
plt.show()

#plotting sample variation with step number
fig, axes = plt.subplots(5, figsize=(8, 8), sharex=True)
labels = [r'$k_{\phi}$', r'$k_X$', r'$g_{3\phi}$', r'$g_{3X}$', r'$g_{4\phi}$']
for i in range(len(labels)):
    ax = axes[i]
    ax.plot(np.reshape(samples[:,i], (900,-1)),color='k', alpha=0.1)
    ax.set_xlim(0, 900)
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel("step number");
plt.show()