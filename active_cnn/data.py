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
    lamost_collection = h5py.File(dataset_file)['lamost_dr2']
    lamost_ids = lamost_collection['filenames'][:]
    lamost_flux = lamost_collection['spectra'][...]
    return lamost_ids, lamost_flux


def save(it, hdf5, ids_tr, X_tr, y_tr, ids, X):
    with h5py.File(hdf5) as f:
        it_gr = f.create_group('iteration_{:02}'.format(it))
        matrixes_names = [(X_tr, 'X_train'), (y_tr, 'y_train'), (X, 'X')]
        for mat, name in matrixes_names:
            dt = it_gr.create_dataset(name, mat.shape, mat.dtype)
            dt[...] = mat

        # variable length data type
        str_dt = h5py.special_dtype(vlen=str)
        vectores_names = [(ids_tr, 'ids_train'), (ids, 'ids')]
        for vec, name in vectores_names:
            dt = it_gr.create_dataset(name, vec.shape, str_dt)
            dt[...] = vec


def renew_datasets(
        X_train, ids_train, y_train, X, ids, index, oracle, iteration
        ):
    X_train = np.concatenate((X_train, X[index]))
    ids_train = np.concatenate((ids_train, ids[index]))
    y_train = np.concatenate((
        y_train, oracle[oracle['iteration'] == iteration]['label'].values.astype(y_train.dtype)
        ))

    bool_index = np.zeros(X.shape[0], np.bool)
    bool_index[index] = True
    X = X[~bool_index]
    ids = ids[~bool_index]
    return X_train, ids_train, y_train, X, ids
