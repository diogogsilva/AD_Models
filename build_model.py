# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:56:52 2020

@author: bruno
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import plot_model

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


'''
Custom RMSE loss function.
To be used when compiling the model.

Parameters
    ----------
    y_true: true values
    y_pred: model's prediction
        
Returns
    -------
    rmse value
'''


def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))


'''
Custom R^2 function (Coefficient of Determination score).
To be used when compiling the model.

Parameters
    ----------
    y_true: true values
    y_pred: model's prediction
        
Returns
    -------
    R^2 value
'''


def coeff_determination(y_true, y_pred):
    ss_res = tf.keras.backend.sum(tf.keras.backend.square(y_true-y_pred))
    ss_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true-tf.keras.backend.mean(y_true)))
    return (1 - ss_res/(ss_tot + tf.keras.backend.epsilon()))


'''
Functional API used to build a generic LSTM model.

Parameters
    ----------
    features: number of features in the input sequence.
    timesteps: number of timesteps that make a sequence.
    layers: number of hidden LSTM layers
    neurons: number of neurons in the layers.
    activation: activation function in the Dense layer.
    dropout_rate: dropout rate at the Dense layer.
    log: print the model's summary and save the model's plot in the filesystem. 
    seed: the random seed.
        
Returns
    -------
    A LSTM model
    (or None if model's validity fails)
'''


def build_lstm_model(features, timesteps=7, layers=2, neurons=64, activation='relu', dropout_rate=0.5, log=False, seed=91195003):
    tf.random.set_seed(seed)
    
    inputs = Input(shape=(timesteps, features))
        
    for i in range(layers):
        #if first LSTM layer
        if i == 0: 
            #if also the last LSTM layer then return_sequences=False
            if i+1 == layers: 
                x = LSTM(int(neurons/2), return_sequences=False)(inputs)
            #it has more layers! So return_sequences=True
            else: 
                x = LSTM(neurons, return_sequences=True)(inputs)
        #if last LSTM layer then return_sequences=False
        elif i+1 == layers: 
            x = LSTM(neurons*2, return_sequences=False)(x)
        #if not the last LSTM layer then return_sequences=True
        else:
            x = LSTM(neurons, return_sequences=True)(x)
    
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    
    lstmModel = Model(inputs=inputs, outputs=outputs, name='lstm_model')
    
    lstmModel.compile(
        loss=rmse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', rmse]
    )    
    
    if log:
        lstmModel.summary()
        #save the model's plot
        from pathlib import Path
        Path('results').mkdir(parents=True, exist_ok=True)
        plot_model(lstmModel, 'results/lstmModel.png', show_shapes=True)
    
    return lstmModel


'''
Functional API used to build a generic MLP model.
Flatten layer is required to flatten the input data.

Parameters
    ----------
    features: number of features in the input sequence.
    timesteps: number of timesteps that make a sequence.
    layers: number of hidden Dense layers.
    neurons: number of neurons in the layers.
    activation: activation function in the Dense layers.
    dropout_rate: dropout rate at the Dense layer.
    log: print the model's summary and save the model's plot in the filesystem. 
    seed: the random seed.
        
Returns
    -------
    A MLP model
    (or None if model's validity fails)
'''


def build_mlp_model(features, timesteps=7, layers=2, neurons=64, activation='relu', dropout_rate=0.5, log=False, seed=91195003):
    tf.random.set_seed(seed)
    
    inputs = Input(shape=(timesteps, features))
    f = Flatten()(inputs)
        
    for i in range(layers):
        #if first Dense layer
        if i == 0: 
            #and also the last
            if i+1 == layers: 
                x = Dense(neurons, activation=activation)(f)
            #it has more layers
            else: 
                x = Dense(neurons/2, activation=activation)(f)
        #if last Dense layer
        elif i+1 == layers: 
            x = Dense(neurons*2, activation=activation)(x)
        #if middle layers
        else:
            x = Dense(neurons, activation=activation)(x)
    
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    
    mlpModel = Model(inputs=inputs, outputs=outputs, name='mlp_model')
    
    mlpModel.compile(
        loss=rmse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', rmse]
    )    
    
    if log:
        mlpModel.summary()
        #save the model's plot
        from pathlib import Path
        Path('results').mkdir(parents=True, exist_ok=True)
        plot_model(mlpModel, 'results/mlpModel.png', show_shapes=True)
    
    return mlpModel


'''
Functional API used to build a generic CNN model.

Parameters
    ----------
    features: number of features in the input sequence.
    timesteps: number of timesteps that make a sequence.
    layers: number of hidden Conv1D layers.
    activation: activation function in Conv1D and Dense layers.
    dropout_rate: dropout rate at the Dense layer.
    filters: number of filters that aim to learn patterns in the sequence
    kernel_size: the size of the kernel applied over the sequence (reduces timesteps as -> (TIMESTEPS-KERNEL)+1)
    pool_size: the size of the pool in the pooling layer (channels_first reduces the filters and not the number of timesteps)
    log: print the model's summary and save the model's plot in the filesystem. 
    seed: the random seed.
        
Returns
    -------
    A CNN model
    (or None if model's validity fails)
'''


def build_cnn_model(features, timesteps=7, layers=2, activation='relu', dropout_rate=0.5, filters=16, kernel_size=3, pool_size=2, log=False, seed=91195003):
    tf.random.set_seed(seed)
    
    ###check model's viability. If false, return None and prevent model generation
    #how will the timesteps decrease from one convolution to another (note that channels_first pooling does not decrease dimensionality of timesteps)
    kernel_stepsize = kernel_size-1
    total_kernel_stepsize = kernel_stepsize*layers
    if timesteps-total_kernel_stepsize <= 0:
        return None
    
    inputs = Input(shape=(timesteps, features))
        
    for i in range(layers):
        #if first CNN layer
        if i == 0: 
            #and also the last
            if i+1 == layers: 
                x = Conv1D(filters=filters/2, kernel_size=kernel_size, activation=activation, data_format='channels_last')(inputs)
                x = AveragePooling1D(pool_size=pool_size, data_format='channels_last')(x)
                x = Flatten()(x)
            #it has more layers
            else: 
                x = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, data_format='channels_last')(inputs)
                x = AveragePooling1D(pool_size=pool_size, data_format='channels_last')(x)
        #if last layer
        elif i+1 == layers: 
            x = Conv1D(filters=filters*2, kernel_size=kernel_size, activation=activation, data_format='channels_last')(x)
            x = AveragePooling1D(pool_size=pool_size, data_format='channels_last')(x)
            x = Flatten()(x)
        #if middle layers
        else:
            x = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, data_format='channels_last')(x)
            x = AveragePooling1D(pool_size=pool_size, data_format='channels_last')(x)
    
    x = Dense(filters)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    
    cnnModel = Model(inputs=inputs, outputs=outputs, name='cnn_model')
    
    cnnModel.compile(
        loss=rmse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', rmse]
    )    
    
    if log:
        cnnModel.summary()
        #save the model's plot
        from pathlib import Path
        Path('results').mkdir(parents=True, exist_ok=True)
        plot_model(cnnModel, 'results/cnnModel.png', show_shapes=True)
    
    return cnnModel


'''
Functional API used to build a generic GRU model.

Parameters
    ----------
    features: number of features in the input sequence.
    timesteps: number of timesteps that make a sequence.
    layers: number of hidden GRU layers
    neurons: number of neurons in the layers.
    activation: activation function in the Dense layer.
    dropout_rate: dropout rate at the Dense layer.
    log: print the model's summary and save the model's plot in the filesystem. 
    seed: the random seed.
        
Returns
    -------
    A GRU model
    (or None if model's validity fails)
'''


def build_gru_model(features, timesteps=7, layers=2, neurons=64, activation='relu', dropout_rate=0.5, log=False, seed=91195003):
    tf.random.set_seed(seed)
    
    inputs = Input(shape=(timesteps, features))
        
    for i in range(layers):
        #if first GRU layer
        if i == 0: 
            #if also the last GRU layer then return_sequences=False
            if i+1 == layers: 
                x = GRU(int(neurons/2), return_sequences=False)(inputs)
            #it has more layers! So return_sequences=True
            else: 
                x = GRU(neurons, return_sequences=True)(inputs)
        #if last GRU layer then return_sequences=False
        elif i+1 == layers: 
            x = GRU(neurons*2, return_sequences=False)(x)
        #if not the last GRU layer then return_sequences=True
        else:
            x = GRU(neurons, return_sequences=True)(x)
    
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    
    gruModel = Model(inputs=inputs, outputs=outputs, name='gru_model')
    
    gruModel.compile(
        loss=rmse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', rmse]
    )    
    
    if log:
        gruModel.summary()
        #save the model's plot
        from pathlib import Path
        Path('results').mkdir(parents=True, exist_ok=True)
        plot_model(gruModel, 'results/gruModel.png', show_shapes=True)
    
    return gruModel


'''
Functional API used to build a custom hybrid LSTM model.

Parameters
    ----------
    features: number of features in the input sequence.
    timesteps: number of timesteps that make a sequence.
    layers: number of hidden LSTM layers.
    neurons: number of neurons in the layers.
    activation: activation function in the Dense layer.
    dropout_rate: dropout rate at the Dense layer.
    filters: number of filters that aim to learn patterns in the sequence.
    kernel_size: the size of the kernel applied over the sequence (reduces timesteps as -> (TIMESTEPS-KERNEL)+1).
    pool_size: the size of the pool in the pooling layer (channels_first reduces the filters and not the number of timesteps).
    log: print the model's summary and save the model's plot in the filesystem. 
    seed: the random seed.
        
Returns
    -------
    A custom hybrid model
    (or None if model's validity fails)
'''


def build_custom_model(features, timesteps=7, neurons=64, activation='relu', dropout_rate=0.5, filters=16, kernel_size=3, pool_size=2, log=False, seed=91195003):
    tf.random.set_seed(seed)
    
    inputs = Input(shape=(timesteps, features))
    
    #sequence recurrent encoding
    x_lstm = LSTM(int(neurons/2), return_sequences=True)(inputs)
    x_lstm = LSTM(neurons, return_sequences=True)(x_lstm)
    x_lstm = LSTM(int(neurons*2), return_sequences=True)(x_lstm)
    x_lstm = Flatten()(x_lstm)
    
    #conv1d sequence encoding
    x_conv = Conv1D(filters=int(filters/2), kernel_size=kernel_size, activation=activation, data_format='channels_last')(inputs)
    x_conv = AveragePooling1D(pool_size=pool_size, data_format='channels_first')(x_conv)
    x_conv = Conv1D(filters=int(filters*2), kernel_size=kernel_size, activation=activation, data_format='channels_last')(x_conv)
    x_conv = AveragePooling1D(pool_size=pool_size, data_format='channels_first')(x_conv)
              
    x_conv_lstm = LSTM(neurons, return_sequences=True)(x_conv)
    x_conv_lstm = LSTM(int(neurons/2), return_sequences=True)(x_conv_lstm)
    x_conv_lstm = Flatten()(x_conv_lstm)
    
    #concatenate
    x_conv = Flatten()(x_conv)
    x = Concatenate(axis=1)([x_lstm, x_conv])
    x = Dense(neurons, activation=activation)(x)
    
    #second concatenation
    x = Concatenate(axis=1)([x_conv_lstm, x])
    
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    
    customModel = Model(inputs=inputs, outputs=outputs, name='custom_model')
    
    customModel.compile(
        loss=rmse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', rmse]
    )    
    
    if log:
        customModel.summary()
        #save the model's plot
        from pathlib import Path
        Path('results').mkdir(parents=True, exist_ok=True)
        plot_model(customModel, 'results/customModel.png', show_shapes=True)
    
    return customModel


def build_custom_model_v2(features, timesteps=7, activation='relu', dropout_rate=0.5, log=False, seed=91195003):
    tf.random.set_seed(seed)
    
    inputs = Input(shape=(timesteps, features))
    
    #sequence recurrent encoding
    x_lstm_1 = LSTM(8, return_sequences=True)(inputs)
    
    x = Concatenate()([inputs, x_lstm_1])
    
    x_lstm_2 = LSTM(16, return_sequences=True)(x)
    x_lstm_2 = LSTM(16, return_sequences=True)(x_lstm_2)
    
    x = Concatenate()([inputs, x_lstm_2])
    
    x_lstm_3 = LSTM(32, return_sequences=True)(x)
    x_lstm_3 = LSTM(32, return_sequences=True)(x_lstm_3)
    x_lstm_3 = LSTM(32, return_sequences=True)(x_lstm_3)
    
    x = Concatenate()([inputs, x_lstm_3])
    x = Flatten()(x)
    
    x = Dense(64, activation=activation)(x)    
    x = Dense(128, activation=activation)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    
    customModel = Model(inputs=inputs, outputs=outputs, name='custom_model_v2')
    
    customModel.compile(
        loss=rmse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', rmse]
    )    
    
    if log:
        customModel.summary()
        #save the model's plot
        from pathlib import Path
        Path('results').mkdir(parents=True, exist_ok=True)
        plot_model(customModel, 'results/customModelv2.png', show_shapes=True)
    
    return customModel


'''
Functional API used to build an explainable LSTM model.

Parameters
    ----------
    features: number of features in the input sequence.
    timesteps: number of timesteps that make a sequence.
    layers: number of hidden LSTM layers
    neurons: number of neurons in the layers.
    activation: activation function in the Dense layer.
    dropout_rate: dropout rate at the Dense layer.
    log: print the model's summary and save the model's plot in the filesystem. 
    seed: the random seed.
        
Returns
    -------
    A XAI LSTM model
    (or None if model's validity fails)
'''


def build_xai_lstm_model(features, timesteps=7, layers=2, neurons=64, activation='relu', dropout_rate=0.5, log=False, seed=91195003):
    tf.random.set_seed(seed)
    
    h_list = list()
    c_list = list()
    
    inputs = Input(shape=(timesteps, features))
        
    for i in range(layers):
        #if first LSTM layer
        if i == 0: 
            #if also the last LSTM layer then return_sequences=False
            if i+1 == layers: 
                x, h, c = LSTM(neurons/2, return_sequences=True, return_state=True)(inputs)
                h_list.append(x)
                c_list.append(c)
            #it has more layers! So return_sequences=True
            else: 
                x, h, c = LSTM(neurons, return_sequences=True, return_state=True)(inputs)
                h_list.append(x)
                c_list.append(c)
        #if last LSTM layer then return_sequences=False
        elif i+1 == layers: 
            x, h, c = LSTM(neurons*2, return_sequences=True, return_state=True)(x)
            h_list.append(x)
            c_list.append(c)
        #if not the last LSTM layer then return_sequences=True
        else:
            x, h, c = LSTM(neurons, return_sequences=True, return_state=True)(x)
            h_list.append(x)
            c_list.append(c)
    
    
    x = Dense(neurons, activation=activation)(h)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    
    lstmModel = Model(inputs=inputs, outputs=outputs, name='lstm_model')
    lstmModel.compile(
        loss=rmse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', rmse]
    )    
    
    lstmModel_prediction = Model(inputs=inputs, outputs=[outputs, h_list, c_list], name='lstm_model_pred')
    lstmModel_prediction.compile(
        loss=rmse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', rmse]
    )    
    
    if log:
        lstmModel.summary()
        #save the model's plot
        from pathlib import Path
        Path('results').mkdir(parents=True, exist_ok=True)
        plot_model(lstmModel, 'results/lstmModel.png', show_shapes=True)
    
    return lstmModel, lstmModel_prediction

def build_if_model(n_estimators=100, contamination='auto', bootstrap=False, log=False):
    ifModel = IsolationForest(n_estimators=n_estimators, contamination=contamination, bootstrap=bootstrap)
    
    if log:
        ifModel.summary()
        #save the model's plot
        from pathlib import Path
        Path('results').mkdir(parents=True, exist_ok=True)
        plot_model(ifModel, 'results/ifModel.png', show_shapes=True)
        
    return ifModel

def build_ocsvm_model(kernel='rbf', gamma=0.001, nu=0.03, log=False):
    ocsvmModel = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    
    if log:
        ocsvmModel.summary()
        #save the model's plot
        from pathlib import Path
        Path('results').mkdir(parents=True, exist_ok=True)
        plot_model(ocsvmModel, 'results/ocsvmModel.png', show_shapes=True)
        
    return ocsvmModel

def build_lstm_ae_model(features=1, timesteps=7, layers=2, neurons=64, activation='relu', dropout_rate=0.5, log=False, seed=91195003):
    tf.random.set_seed(seed)
    
    lstmModel = Sequential(name='lstm_ae_model')
    lstmModel.add(LSTM(units=neurons, input_shape=(timesteps, features)))
    lstmModel.add(Dropout(rate=dropout_rate))
    lstmModel.add(RepeatVector(n=timesteps))
    lstmModel.add(LSTM(units=neurons, return_sequences=True))
    lstmModel.add(Dropout(rate=dropout_rate))
    lstmModel.add(TimeDistributed(Dense(features)))
    
    '''inputs = Input(shape=(timesteps, features))
        
    for i in range(layers):
        #if first LSTM layer
        if i == 0: 
            #if also the last LSTM layer then return_sequences=False
            if i+1 == layers: 
                x = LSTM(int(neurons/2), return_sequences=False)(inputs)
            #it has more layers! So return_sequences=True
            else: 
                x = LSTM(neurons, return_sequences=True)(inputs)
        #if last LSTM layer then return_sequences=False
        elif i+1 == layers: 
            x = LSTM(neurons*2, return_sequences=False)(x)
        #if not the last LSTM layer then return_sequences=True
        else:
            x = LSTM(neurons, return_sequences=True)(x)

    x = Dropout(dropout_rate)(x)
    x = RepeatVector(n=timesteps)(x)
    
    for i in range(layers):
        #if first LSTM layer
        if i == 0: 
            #if also the last LSTM layer then return_sequences=False
            if i+1 == layers: 
                x = LSTM(int(neurons/2), return_sequences=False)(inputs)
            #it has more layers! So return_sequences=True
            else: 
                x = LSTM(neurons, return_sequences=True)(inputs)
        #if last LSTM layer then return_sequences=False
        elif i+1 == layers: 
            x = LSTM(neurons*2, return_sequences=False)(x)
        #if not the last LSTM layer then return_sequences=True
        else:
            x = LSTM(neurons, return_sequences=True)(x)
    
    x = Dropout(dropout_rate)(x)
    print(x)
    outputs = TimeDistributed(Dense(features))(x)
    
    lstmModel = Model(inputs=inputs, outputs=outputs, name='lstm_ae_model')'''
    
    lstmModel.compile(
        loss=rmse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', rmse]
    )    
    
    return lstmModel