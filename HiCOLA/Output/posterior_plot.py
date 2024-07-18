import numpy as np
import corner
import matplotlib.pyplot as plt

samples = np.loadtxt('samples.txt')
probabilities = np.loadtxt('probabilities.txt')
z, E_LCDM, best_fit_model, med_model, spread = np.loadtxt('posterior-input.txt', unpack=True)

theta_max = samples[np.argmax(probabilities)]

plt.plot(z, E_LCDM, label='LCDM')
plt.plot(z, best_fit_model, label='highest likelihood')
plt.fill_between(z, med_model-spread, med_model+spread, color='grey', alpha=0.5, label=r'$1\sigma$ Posterior Spread')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('z')
plt.ylabel('E')
plt.xlim(right = 10)
plt.ylim(top = 10)
plt.show()

plt.plot(z, best_fit_model/E_LCDM, label='Highest likelihood model')
plt.plot(z, med_model/E_LCDM, label='Median model')
plt.fill_between(z, (med_model-spread)/E_LCDM, (med_model+spread)/E_LCDM, color='grey', alpha=0.5, label=r'$1\sigma$ Posterior Spread')
plt.legend()
plt.legend()
plt.xscale('log')
plt.xlabel('z')
plt.ylabel('E/E_LCDM')
plt.show()

labels = ['k_phi', 'k_X', 'g_3phi', 'g_3X', 'g_4phi']
corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16,0.5,0.84])
plt.show()
