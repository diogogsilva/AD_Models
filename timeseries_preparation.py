# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 22:42:15 2020

@author: bruno
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


'''
Normalize the data.
/!\ Updates the given dataframe.

Parameters
    ----------
    df: the dataframe with a set of features
    norm_range: the range to be used to normalize the data
        
Returns
    -------
    a dictionary containing the used scalers per feature
'''
def data_normalization(df, norm_range=(-1, 1)):
    dict_of_scalers = dict()
    for col in df.columns:
        if df[col].dtype != type(object):
            scaler = MinMaxScaler(feature_range=norm_range)
            df[[col]] = scaler.fit_transform(df[[col]])
            dict_of_scalers[col] = scaler
    return dict_of_scalers


'''
Create a supervised problem from a timeseries.
Make sure that the target feature is called 'value'.

Parameters
    ----------
    df: the dataframe with a set of features
    timesteps: the number of timesteps that make an input sequence (the X)
        
Returns
    -------
    (data, target) or - if you prefer - (X, y)
'''
def to_supervised(df, timesteps):    
    data = df.values
    col_target_position = df.columns.get_loc('value')
    X, y = list(), list()
    #iterate over the training set to create X and y
    dataset_size = len(data)
    for curr_pos in range(dataset_size-timesteps):
        #end of the input sequence is the current position + the number of timesteps of the input sequence
        input_index = curr_pos + timesteps
        #end of the labels corresponds to the end of the input sequence + 1
        label_index = input_index + 1
        X.append(data[curr_pos:input_index, :])
        y.append(data[input_index:label_index, col_target_position])
    #using np.float32 for GPU performance
    return np.array(X).astype('float32'), np.array(y).astype('float32')


'''
Split data into training and validation.

Parameters
    ----------
    training: all training indexes
    perc: percentage of validation data
        
Returns
    -------
    (train_idx, val_idx) -> training and validation indexes
'''
def split_dataset(training, perc=10):
    train_idx = np.arange(0, int(len(training)*(100-perc)/100))
    val_idx = np.arange(int(len(training)*(100-perc)/100+1), len(training))
    return train_idx, val_idx