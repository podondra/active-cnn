import h5py
import pandas as pd


def get_ondrejov_dataset(dataset_file):
    ondrejov_dataset = pd.read_csv(dataset_file)
    # first 12 columns are metadata
    ondrejov_ids = ondrejov_dataset['id'].values
    ondrejov_flux = ondrejov_dataset.iloc[:, 12:].values
    ondrejov_labels = ondrejov_dataset['label'].values
    return ondrejov_ids, ondrejov_flux, ondrejov_labels


def get_lamost_dataset(dataset_file):
    lamost_collection = h5py.File(dataset_file)['lamost_dr2']
    lamost_ids = lamost_collection['filenames'][:]
    lamost_flux = lamost_collection['spectra'][...]
    return lamost_ids, lamost_flux
