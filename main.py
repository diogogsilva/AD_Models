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
import tensorflow as tf

'''
Just load the dataset
'''
df_ph = ld.load_ph_dataset(univariate=True, colab=COLAB)

'''
Normalize it
'''
df_data = df_ph.copy()
scalers = tsp.data_normalization(df_data, norm_range=(0, 1))

'''
Set the TimeSeries parameters
'''
multisteps = 2
cv_splits = 3
#must respect the name of the arguments of the build_model function
#timesteps are mandatory
hyperparameters = {
    'layers': [1, 2],
    'neurons': [32, 64, 128],
    'dropout_rate': [0.0, 0.5],
    'activation': ['relu', 'tanh'],
    'timesteps': [7, 14, 21],
    'batch_size': [10, 20, 30],
    'filters': [32, 64],
    'kernel_size': [4, 5, 6],
    'pool_size': [1, 2],
    'n_estimators': [50, 100, 150, 200],
    'contamination': ['auto', 0.05, 0.1],
    'bootstrap': [True, False]
}
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=25, min_lr=0.00005),
             tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, mode='min', verbose=0, patience=40)]

'''
Learning Curves Analysis
mode=0 -> individually plot loss per cv split
mode=1 -> aggregate loss of all cv splits
'''
### LSTMs
#exp.learning_curves_analysis(df_data, hyperparameters, bm.build_lstm_model, cv_splits, mode=1, verbose=1, epochs_list=[15,150], seed=SEED, colab=COLAB)
### MLPs
#exp.learning_curves_analysis(df_data, hyperparameters, bm.build_mlp_model, cv_splits, mode=1, verbose=1, epochs_list=[150, 250], colab=COLAB)
### CNNs
#exp.learning_curves_analysis(df_data, hyperparameters, bm.build_cnn_model, cv_splits, mode=1, verbose=1, epochs_list=[15, 150], seed=SEED, colab=COLAB)
### GRUs
#exp.learning_curves_analysis(df_data, hyperparameters, bm.build_gru_model, cv_splits, mode=1, verbose=1, epochs_list=[15, 250], colab=COLAB)


### LSTMs
#exp.tune_model(df_data, hyperparameters, cv_splits, multisteps, scalers, bm.build_lstm_model, epochs=25, verbose=0, callbacks=callbacks, resume_at_experiment=1, store_exp_ratio=10, seed=SEED, colab=COLAB)
#exp.tune_model(df_data, hyperparameters, cv_splits, multisteps, scalers, bm.build_lstm_model, epochs=25, verbose=0, callbacks=callbacks, resume_at_experiment=210, store_exp_ratio=3, seed=SEED, colab=COLAB)
### MLPs
#exp.tune_model(df_data, hyperparameters, cv_splits, multisteps, scalers, bm.build_mlp_model, epochs=100, verbose=0, callbacks=callbacks, resume_at_experiment=1, store_exp_ratio=10, colab=COLAB)
### CNNs
#exp.tune_model(df_data, hyperparameters, cv_splits, multisteps, scalers, bm.build_cnn_model, epochs=25, verbose=0, callbacks=callbacks, resume_at_experiment= 440, store_exp_ratio=50, seed=SEED, colab=COLAB)
#exp.tune_model(df_data, hyperparameters, cv_splits, multisteps, scalers, bm.build_cnn_model, epochs=25, verbose=0, callbacks=callbacks, resume_at_experiment= 430, store_exp_ratio=2, seed=SEED, colab=COLAB)
### GRUs
#exp.tune_model(df_data, hyperparameters, cv_splits, multisteps, scalers, bm.build_gru_model, epochs=50, verbose=0, callbacks=callbacks, resume_at_experiment=190, store_exp_ratio=10, seed=SEED, colab=COLAB)
#exp.tune_model(df_data, hyperparameters, cv_splits, multisteps, scalers, bm.build_gru_model, epochs=50, verbose=0, callbacks=callbacks, resume_at_experiment=210, store_exp_ratio=3, seed=SEED, colab=COLAB)
### IFs
exp.tune_model_anomaly_detection(df_data, hyperparameters, cv_splits, multisteps, scalers, bm.build_if_model, verbose=0, callbacks=callbacks, resume_at_experiment=1, store_exp_ratio=2, seed=SEED, colab=COLAB)