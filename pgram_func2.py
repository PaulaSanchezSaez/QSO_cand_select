#by Constanza Yovaniniz
import numpy as np
from astroML.time_series import \
    lomb_scargle, lomb_scargle_BIC, lomb_scargle_bootstrap

def frequency_grid(Pmin, Pmax):
    #generate an angular frequency grid between Pmin and Pmax (assumed to be in days)
    freq_min = 2*np.pi / Pmin
    freq_max = 2*np.pi / Pmax
    return np.linspace(freq_min, freq_max, 1000)

def LS_peak_to_period(omegas, P_LS):
    #find the highest peak in the LS periodogram and return the corresponding period.
    max_freq = omegas[np.argmax(P_LS)]
    return 2*np.pi/max_freq

def get_period_sigf(jd,mag,mag_e): #parameters: time, mag, mag error
	omega = frequency_grid(0.1,1850) # grid of frequencies
	p = lomb_scargle(jd,mag,mag_e,omega,generalized=True) # Lomb-Scargle power associated to omega
	peak = max(p) #power associated with best_period
	best_period = LS_peak_to_period(omega,p) #estimates a period from periodogram

	# Get significance via bootstrap
	D = lomb_scargle_bootstrap(jd,mag,mag_e,omega,generalized=True,N_bootstraps=1000,random_state=0)
	sig1, sig5 = np.percentile(D, [99, 95]) # 95% and 99% confidence, to compare with peak

	return best_period, peak, sig5, sig1
	#period, power(period), cut 95%, cut 99%
