from . import data
import h5py
import pandas as pd


def initialize(ids_tr, X_tr, y_tr, ids, X):
    # create table for estimation of performance
    est_perf_df = pd.DataFrame(columns=('correct', 'iteration'))
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
