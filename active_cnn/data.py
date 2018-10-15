import h5py
import numpy as np
import pandas as pd


def get_ondrejov_dataset(dataset_file):
    ondrejov_dataset = pd.read_csv(dataset_file)
    # first 12 columns are metadata
    ondrejov_ids = ondrejov_dataset['id'].values
    ondrejov_flux = ondrejov_dataset.iloc[:, 12:].values
    ondrejov_labels = ondrejov_dataset['label'].values
    # convert labels to one-hot encoding
    ondrejov_y = np.zeros_like(ondrejov_labels, dtype='int')
    idx = (ondrejov_labels == 'emission') | (ondrejov_labels == 'double-peak')
    ondrejov_y[idx] = 1
    return ondrejov_ids, ondrejov_flux, ondrejov_labels, ondrejov_y


def get_lamost_dataset(dataset_file):
    lamost_collection = h5py.File(dataset_file)['lamost_dr2']
    lamost_ids = lamost_collection['filenames'][:]
    lamost_flux = lamost_collection['spectra'][...]
    return lamost_ids, lamost_flux


def save_data(hdf5, iteration, X_train, y_train, ids_train, X, ids):
    f = h5py.File(hdf5)
    # create group
    group = f.create_group('iteration_{:02}'.format(iteration))
    # variable length data type
    str_dt = h5py.special_dtype(vlen=str)
    # save all matrices
    X_train_dt = group.create_dataset('X_train', X_train.shape, X_train.dtype)
    X_train_dt[...] = X_train
    y_train_dt = group.create_dataset('y_train', y_train.shape, y_train.dtype)
    y_train_dt[...] = y_train
    ids_train_dt = group.create_dataset('ids_train', ids_train.shape, str_dt)
    ids_train_dt[...] = ids_train
    X_dt = group.create_dataset('X', X.shape, X.dtype)
    X_dt[...] = X
    ids_dt = group.create_dataset('ids', ids.shape, str_dt)
    ids_dt[...] = ids
    f.close()


def renew_datasets(
        X_train, ids_train, y_train, X, ids, index, oracle, iteration
        ):
    X_train = np.concatenate((X_train, X[index]))
    ids_train = np.concatenate((ids_train, ids[index]))
    y_train = np.concatenate((
        y_train, oracle[oracle['iteration'] == iteration]['label'].values
        ))

    bool_index = np.zeros(X.shape[0], np.bool)
    bool_index[index] = True
    X = X[~bool_index]
    ids = ids[~bool_index]
    return X_train, ids_train, y_train, X, ids
