# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:41:22 2021

@author: diogo.s.a
"""

COLAB = False

SEED = 91195003

import load_dataset as ld
import timeseries_preparation as tsp
import experiments as exp
import build_model as bm

'''
Just load the dataset
'''
#df = ld.load_ph_dataset(univariate=True, colab=COLAB)
df = ld.load_amonia_dataset(univariate=False, colab=COLAB)

'''
Normalize it
'''
#df_data = df_ph.copy()
df_data = df.copy()
scalers = tsp.data_normalization(df_data, norm_range=(0, 1))

cv_splits = 3

hyperparameters = {
    #LSTM AE
    'layers': [1],
    'neurons': [32, 64, 128],
    'dropout_rate': [0.0, 0.5],
    'activation': ['relu', 'tanh'],
    'timesteps': [1],
    'batch_size': [10, 20, 30],
    #iF
    'n_estimators': [100, 150, 200],
    'contamination': [0.1, 0.005],
    'bootstrap': [True],
    #OCSVM
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.003],
    'nu': [0.001, 0.01, 0.5]
}

best_hyperparams = {
    #LSTM_AE
    'layers': [1], 
    'neurons': [32], 
    'dropout_rate': [0.0], 
    'activation': ['relu'], 
    'timesteps': [1], 
    'batch_size': [10],
    #iF
    'n_estimators': [200], 
    'contamination': [0.005], 
    'bootstrap': [True],
    #OCSVM
    'kernel': ['rbf'], 
    'gamma': ['auto'], 
    'nu': [0.001]
}

#exp.anomaly_detection_models(df_data, hyperparameters, cv_splits, scalers, bm.build_if_model, epochs=-1, batch_size=-1, resume_at_experiment=1, store_exp_ratio=72, seed=SEED, colab=COLAB)
#exp.anomaly_detection_models(df_data, hyperparameters, cv_splits, scalers, bm.build_ocsvm_model, epochs=-1, batch_size=-1, resume_at_experiment=1, store_exp_ratio=72, seed=SEED, colab=COLAB)
exp.anomaly_detection_models(df_data, hyperparameters, 2, scalers, bm.build_lstm_ae_model, epochs=100, batch_size=10, resume_at_experiment=1, store_exp_ratio=12, seed=SEED, colab=COLAB)

def get_best_AD_model(algorithm = 'lstm_ae'):
    if(algorithm == 'if'):
        return exp.anomaly_detection_models(df_data, best_hyperparams, cv_splits, scalers, bm.build_if_model, epochs=-1, batch_size=-1, resume_at_experiment=1, store_exp_ratio=9999, seed=SEED, colab=COLAB, silent=True)
    elif(algorithm == 'ocsvm'):
        return exp.anomaly_detection_models(df_data, best_hyperparams, cv_splits, scalers, bm.build_ocsvm_model, epochs=-1, batch_size=-1, resume_at_experiment=1, store_exp_ratio=9999, seed=SEED, colab=COLAB, silent=True)
    elif(algorithm == 'lstm_ae'):
        return exp.anomaly_detection_models(df_data, best_hyperparams, cv_splits, scalers, bm.build_lstm_ae_model, epochs=100, batch_size=10, resume_at_experiment=1, store_exp_ratio=9999, seed=SEED, colab=COLAB, silent=True)
    else:
        print('ERROR: Undefined model...')
        return None