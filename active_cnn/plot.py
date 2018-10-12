import numpy as np


def plot_ondrejov_spectrum(spectrum_id, ondrejov, ax):
    wavelengths = np.linspace(6519, 6732, num=140)
    index = ondrejov[0] == spectrum_id
    fluxes = ondrejov[1][index].reshape(140)
    ax.plot(wavelengths, fluxes)
