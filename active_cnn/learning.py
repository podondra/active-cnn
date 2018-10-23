from functools import partial
from . import data
import h5py
import numpy as np
import pandas as pd


def initialize(ids_tr, X_tr, y_tr, ids, X):
    # create table for estimation of performance
    est_perf_df = pd.DataFrame(columns=('predicted', 'correct', 'iteration'))
    est_perf_df.index.name = 'identifier'
    # create table for storage of oracle labelling
    oracle_df = pd.DataFrame(columns=('label', 'iteration'))
    oracle_df.index.name = 'identifier'
    # build a dictionary will of data for learning
    data_dict = {
            'it': 0,
            'hdf5': 'data/learning-data.h5',
            'ids_tr': ids_tr,
            'X_tr': X_tr,
            'y_tr': y_tr,
            'ids': ids,
            'X': X,
            'est_perf_df': est_perf_df,
            'oracle_df': oracle_df,
            }
    h5py.File(data_dict['hdf5'], 'w').close()
    return data_dict


def save(data_dict):
    data.save(
            it=data_dict['it'],
            hdf5=data_dict['hdf5'],
            ids_tr=data_dict['ids_tr'],
            X_tr=data_dict['X_tr'],
            y_tr=data_dict['y_tr'],
            ids=data_dict['ids'],
            X=data_dict['X'],
            )
    # back up DataFrames
    data_dict['est_perf_df'].to_csv('data/estimated-performance.csv')
    data_dict['oracle_df'].to_csv('data/oracle-labels.csv')


def load(data_dict):
    # TODO
    # y_pred = f['iteration_{:02}/y_pred'.format(iteration)][...]
    ...


def show_prediction_stats(data_dict):
    return np.unique(data_dict['labels'], return_counts=True)


def get_random_spectra(data_dict):
    # TODO
    # emission_index = np.arange(labels.shape[0])[labels]
    # random_index = np.random.choice(emission_index, size=32, replace=False)
    # gen = (filename for filename in ids[random_index])
    ...


def mark_spectrum(identifier, data_dict, predicted_label, correct_label):
    row = pd.Series({
        'correct': correct_label,
        'predicted': predicted_label,
        'iteration': data_dict['it']
        }, name=identifier)
    data_dict['est_perf_df'] = data_dict['est_perf_df'].append(row)


mark_not_interesting = partial(mark_spectrum, correct_label='not-interesting')
mark_emission = partial(mark_spectrum, correct_label='emission')
mark_double_peak = partial(mark_spectrum, correct_label='double-peak')


def get_reclassification_spectra(data_dict):
    # TODO
    # distance = np.abs(y_pred - 0.5)
    # index = np.argsort(distance)[:64]
    # gen = (filename for filename in ids[index])
    ...


def classify_spectrum():
    # TODO
    # row = pd.Series({'label': 0, 'iteration': iteration}, name=filename)
    # oracle = oracle.append(row)
    ...


# TODO
clasify_not_interesting = None
clasify_emission = None
clasify_double_peak = None


def next_iteration(data_dict):
    data_dict['it'] += 1
    # TODO
    # data.renew_datasets()
    ...


def save_candidates(data_dict):
    # TODO
    # candidates_filenames = set(ids[labels]) | set(oracle[oracle['label'] == 1].index)
    # len(candidates_filenames)

    # candidates = pd.DataFrame(columns=('path', 'label'))
    # for filename in candidates_filenames:
    #     directory = filename.split('-')[2].split('_sp')[0]
    #     filepath = '/lamost/' + directory + '/' + filename
    #     row = pd.Series({'path': filepath, 'label': 'interesting'})
    #     candidates = candidates.append(row, ignore_index=True)

    # candidates.to_csv('data/first-active-learning-candidates.csv', index=False)
    ...
