import h5py
import numpy as np
import pandas as pd


def get_ondrejov_dataset(dataset_file):
    ondrejov_dataset = pd.read_csv(dataset_file)
    # first 12 columns are metadata
    ondrejov_ids = ondrejov_dataset['id'].values
    ondrejov_flux = ondrejov_dataset.iloc[:, 12:].values
    ondrejov_labels = ondrejov_dataset['label'].values
    # convert labels to numerical values
    ondrejov_y = np.zeros_like(ondrejov_labels, dtype='int')
    ondrejov_y[ondrejov_labels == 'emission'] = 1
    ondrejov_y[ondrejov_labels == 'double-peak'] = 2
    return ondrejov_ids, ondrejov_flux, ondrejov_labels, ondrejov_y


def get_lamost_dataset(dataset_file):
    with h5py.File(dataset_file, 'r') as f:
        lamost_collection = f['lamost_dr2']
        lamost_ids = lamost_collection['filenames'][...]
        lamost_flux = lamost_collection['spectra'][...]
    return lamost_ids, lamost_flux


def save(it, hdf5, ids_tr, X_tr, y_tr, ids, X):
    it_gr = hdf5.create_group('iteration_{:02}'.format(it))
    matrixes_names = [(X_tr, 'X_tr'), (y_tr, 'y_tr'), (X, 'X')]
    for mat, name in matrixes_names:
        dt = it_gr.create_dataset(name, mat.shape, mat.dtype)
        dt[...] = mat

    # variable length data type
    str_dt = h5py.special_dtype(vlen=str)
    vectores_names = [(ids_tr, 'ids_tr'), (ids, 'ids')]
    for vec, name in vectores_names:
        dt = it_gr.create_dataset(name, vec.shape, str_dt)
        dt[...] = vec
