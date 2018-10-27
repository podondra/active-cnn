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
    oracle_df = pd.DataFrame(columns=('label', 'iteration'))
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
