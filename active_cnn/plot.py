import matplotlib.pyplot as plt
import numpy as np
from spectraml import lamost


def plot_ondrejov_spectrum(spectrum_id, ondrejov, ax):
    wavelengths = np.linspace(6519, 6732, num=140)
    index = ondrejov[0] == spectrum_id
    fluxes = ondrejov[1][index].reshape(140)
    ax.plot(wavelengths, fluxes)


def preview_lamost_spectrum(filename):
    directory = filename.split('-')[2].split('_sp')[0]
    filepath = '/lamost/fits/' + directory + '/' + filename
    name, wave, flux = lamost.read_spectrum(filepath)
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.axvline(x=6564.614, c='k', alpha=0.5, ls='dashed')
    ax1.plot(wave, flux)
    index = (6519 < wave) & (wave < 6732)
    ax2.axvline(x=6564.614, c='k', alpha=0.5, ls='dashed')
    ax2.plot(wave[index], flux[index])
    plt.show()
    plt.close(fig)
