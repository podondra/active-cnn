from functools import partial
from . import data
from . import model
from . import preprocessing
import h5py
import numpy as np
import pandas as pd
from scipy.stats import entropy


def initialize(ids_tr, X_tr, y_tr, ids, X):
    it = 0
    # create table for estimation of performance
    est_perf_df = pd.DataFrame(columns=('predicted', 'correct', 'iteration'))
    est_perf_df.index.name = 'identifier'
    # create table for storage of oracle labelling
    oracle_df = pd.DataFrame(columns=('predicted', 'correct', 'iteration'))
    oracle_df.index.name = 'identifier'
    # create HDF5 file with data
    hdf5 = h5py.File('data/learning-data.h5', 'w')
    data.save(it, hdf5, ids_tr, X_tr, y_tr, ids, X)
    it_gr = hdf5['iteration_{:02}'.format(it)]
    # build a dictionary will of data for learning
    data_dict = {
            'it': it,
            'hdf5': hdf5,
            'ids_tr': it_gr['ids_tr'],
            'X_tr': it_gr['X_tr'],
            'y_tr': it_gr['y_tr'],
            'ids': it_gr['ids'],
            'X': it_gr['X'],
            'est_perf_df': est_perf_df,
            'oracle_df': oracle_df,
            }
    return data_dict


def save_predictions(data_dict, y_pred):
    it = data_dict['it']
    it_gr = data_dict['hdf5']['iteration_{:02}'.format(it)]
    # probabilities
    dt_pred = it_gr.create_dataset('y_pred', y_pred.shape, y_pred.dtype)
    dt_pred[...] = y_pred
    data_dict['y_pred'] = dt_pred
    # labels
    dt_labels = it_gr.create_dataset('labels', (y_pred.shape[0], ), np.int)
    dt_labels[...] = np.argmax(y_pred, axis=1)
    data_dict['labels'] = dt_labels
    # entropy
    dt_entropy = it_gr.create_dataset('entropy', (y_pred.shape[0], ), np.float)
    dt_entropy[...] = entropy(y_pred.T)
    data_dict['entropy'] = dt_entropy


def learning(data_dict):
    X_tr, y_tr = data_dict['X_tr'], data_dict['y_tr']
    X = data_dict['X']
    X_tr_bal, y_tr_bal = preprocessing.balance(X_tr, y_tr)
    # build the model and train it
    cnn = model.get_model()
    model.train(cnn, X_tr_bal, y_tr_bal)
    # save network's weights
    cnn.save('data/cnn-bu-it-{:02}.h5'.format(data_dict['it']))
    # get and save predictions
    y_pred = model.predict(cnn, X)
    save_predictions(data_dict, y_pred)
    return data_dict


def show_prediction_stats(data_dict):
    return np.unique(data_dict['labels'], return_counts=True)


def get_random_spectra(data_dict, size=30):
    labels = data_dict['labels'][:]
    ids = data_dict['ids'][:]
    # take only positive examples: emissions and double-peaks
    idx = np.arange(labels.shape[0])[labels != 0]
    rnd_idx = np.random.choice(idx, size=size, replace=False)
    # return a generator
    return (pair for pair in zip(ids[rnd_idx], labels[rnd_idx]))


def mark_spectrum(identifier, data_dict, predicted_label, correct_label):
    label = ['not-interesting', 'emission', 'double-peak'][predicted_label]
    row = pd.Series({
        'correct': correct_label,
        'predicted': label,
        'iteration': data_dict['it']
        }, name=identifier)
    data_dict['est_perf_df'] = data_dict['est_perf_df'].append(row)


mark_not_interesting = partial(mark_spectrum, correct_label='not-interesting')
mark_emission = partial(mark_spectrum, correct_label='emission')
mark_double_peak = partial(mark_spectrum, correct_label='double-peak')


def get_reclassification_spectra(data_dict, size=100):
    idx = np.argsort(data_dict['entropy'])[-size:]
    data_dict['idx'] = idx
    ids = data_dict['ids'][:]
    labels = data_dict['labels'][:]
    return (pair for pair in zip(ids[idx], labels[idx]))


def classify_spectrum(identifier, data_dict, predicted_label, correct_label):
    label = ['not-interesting', 'emission', 'double-peak'][predicted_label]
    row = pd.Series({
        'correct': correct_label,
        'predicted': label,
        'iteration': data_dict['it']
        }, name=identifier)
    data_dict['oracle_df'] = data_dict['oracle_df'].append(row)


classify_not_interesting = partial(
        classify_spectrum, correct_label='not-interesting'
        )
classify_emission = partial(classify_spectrum, correct_label='emission')
classify_double_peak = partial(classify_spectrum, correct_label='double-peak')


def next_iteration(data_dict):
    it = data_dict['it']
    # store DataFrames
    data_dict['est_perf_df'].to_csv('data/est-perf-df.csv')
    oracle_df = data_dict['oracle_df']
    oracle_df.to_csv('data/oracle-df.csv')
    # modify datasets
    idx = data_dict['idx']
    ids_tr = data_dict['ids_tr']
    X_tr = data_dict['X_tr']
    y_tr = data_dict['y_tr']
    X = data_dict['X'][...]
    ids = data_dict['ids'][:]
    X_tr = np.concatenate((X_tr, X[idx]))
    ids_tr = np.concatenate((ids_tr, ids[idx]))
    # extract labels
    mapping = {'not-interesting': 0, 'emission': 1, 'double-peak': 2}
    y = oracle_df[oracle_df['iteration'] == it]['correct'].map(mapping).values
    y_tr = np.concatenate((y_tr, y))
    # remove reclassified spectra
    bool_idx = np.ones(X.shape[0], np.bool)
    bool_idx[idx] = False
    X = X[bool_idx]
    ids = ids[bool_idx]
    # save datasets
    data_dict['it'] += 1
    it = data_dict['it']
    hdf5 = data_dict['hdf5']
    data.save(it, hdf5, ids_tr, X_tr, y_tr, ids, X)
    # modify data_dict
    it_gr = hdf5['iteration_{:02}'.format(it)]
    data_dict['ids_tr'] = it_gr['ids_tr']
    data_dict['X_tr'] = it_gr['X_tr']
    data_dict['y_tr'] = it_gr['y_tr']
    data_dict['ids'] = it_gr['ids']
    data_dict['X'] = it_gr['X']
    return data_dict


def save_candidates(data_dict):
    ids = data_dict['ids']
    labels = data_dict['labels']
    oracle_df = data_dict['oracle_df']
    # candidates from oracle
    oracle_idx = oracle_df['correct'] != 'not-interesting'
    oracle_cans = oracle_df[oracle_idx].index.values
    oracle_labels = oracle_df[oracle_idx]['correct'].values
    # cnn's candidates
    cans_idx = labels[:] != 0
    cans_filenames = ids[cans_idx]
    cans_num_labels = labels[cans_idx]
    cans_labels = np.zeros_like(cans_filenames, dtype=oracle_labels.dtype)
    cans_labels[cans_num_labels == 1] = 'emission'
    cans_labels[cans_num_labels == 2] = 'double-peak'
    # merge
    all_filenames = np.concatenate((cans_filenames, oracle_cans))
    all_labels = np.concatenate((cans_labels, oracle_labels))
    # DataFrame for candidates
    cans_df = pd.DataFrame(columns=('path', 'label'))
    for filename, label in zip(all_filenames, all_labels):
        directory = filename.split('-')[2].split('_sp')[0]
        filepath = '/lamost/' + directory + '/' + filename
        row = pd.Series({'path': filepath, 'label': label})
        cans_df = cans_df.append(row, ignore_index=True)
    cans_df.to_csv('candidates.csv', index=False)

def finalize(data_dict):
    data_dict['hdf5'].close()
