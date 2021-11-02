# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:18:23 2020

@author: bruno
"""

import pandas as pd
import numpy as np
import timeseries_preparation as tsp
import itertools
import inspect
import time
import plots
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
#import shap
    

'''
Analyse if overfitting is present for the lighter/heaviest model.
Stores pdf files with images for each epochs in epochs_list 
for each lighter and heavier combination of hyperparameters

Parameters
    ----------
    df: the dataframe holding the data
    hyperparameters: a dictionary of hyperparameters to tune (pay attention to the dict keys - timesteps mandatory)
    build_model_function: the function to be used to build a deep learning model
    cv_splits: number of splits to be used in the TimeSeries Cross Validator
    mode: 0 for plotting splits individually; 1 for plotting averaged losses for all splits
    verbose: the verbose mode
    epochs_list: experiment these amount of epochs
    batch_size: default batch size if not provided in the hyperparameters
    seed: random seed to be used
    colab: writing to drive?    
'''


def learning_curves_analysis(df, hyperparameters, build_model_function, cv_splits, mode=0, verbose=0, epochs_list=[100, 500], batch_size=32, seed=91195003, colab=False):
    #obtain prepared param list and names
    param_list, param_names = prepare_param_list(hyperparameters, build_model_function)
    try:
        timesteps_position = param_names.index('timesteps')
    except:
        print("Error: You MUST tune the timesteps!")
        raise
    lighter_param_comb = param_list[0]
    heavier_param_comb = param_list[-1]
    #get the batch_size if the user sets it. Otherwise, use default value (which can be passed as argument)
    if 'batch_size' in param_names:
        lighter_batch_size = lighter_param_comb[param_names.index('batch_size')]
        heavier_batch_size = heavier_param_comb[param_names.index('batch_size')]
    else:
        lighter_batch_size = batch_size
        heavier_batch_size = batch_size
    #plot the learning curves
    plot_learning_curve(df, timesteps_position, epochs_list, cv_splits, build_model_function, param_names, lighter_param_comb, lighter_batch_size, verbose, 'lighter_combination', mode, seed, colab)
    plot_learning_curve(df, timesteps_position, epochs_list, cv_splits, build_model_function, param_names, heavier_param_comb, heavier_batch_size, verbose, 'heavier_combination', mode, seed, colab)


'''
Fitting and ploting the learning curves for each epochs in epochs_list and for each split

Parameters
    ----------
    df: the dataframe holding the data
    timesteps_position: position of the timesteps feature in the param_comb
    epochs_list: experiment these amount of epochs
    cv_splits: number of splits to be used in the TimeSeries Cross Validator
    build_model_function: the function to be used to build a deep learning model
    param_names: the hyperparameters' names
    param_combination: the combination of values
    batch_size: default batch size if not provided in the hyperparameters
    verbose: the verbose mode
    plot_title_note: indentify if it is the lighter or heavier combination of hyperparameters
    mode: 0 for plotting splits individually; 1 for plotting averaged losses for all splits
    seed: random seed to be used
    colab: writing to drive?    
''' 
def plot_learning_curve(df, timesteps_position, epochs_list, cv_splits, build_model_function, param_names, param_comb, batch_size, verbose, plot_title_note, mode, seed, colab):
    #create the supervised problem
    X, y = tsp.to_supervised(df, param_comb[timesteps_position])
    for epochs in epochs_list:
        #storing history values
        hist_list = list()
        #timeseries split for model validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        for train_index, _ in tscv.split(X):
            #further split into training and validation sets
            train_idx, val_idx = tsp.split_dataset(train_index, perc=30)
            #build data
            X_train, y_train = X[train_idx], y[train_idx] 
            X_val, y_val = X[val_idx], y[val_idx] 
            #generically build model
            features = np.size(X_train, 2)  #the last dimension
            model = build_model(features, param_names, param_comb, build_model_function, seed)
            if model is None:
                print('Cannot plot learning curve for these hyperparameters (Model skipped as it is not valid):')
                log_hyperparameters(param_names, param_comb)
                return
            #fit
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, shuffle=False, verbose=verbose)
            #store history results
            hist_list.append(history)
        #plot learning curves
        loss_metric_name = model.metrics_names[0]
        if mode == 0:
            plots.plot_learning_curve_per_split(hist_list, loss_metric_name, plot_title_note, model.name, colab)
        else:
            mean_training_loss = np.mean([hist.history['loss'] for hist in hist_list], axis=0)
            mean_validation_loss = np.mean([hist.history['val_loss'] for hist in hist_list], axis=0)
            plots.plot_learning_curve_aggregated_splits(mean_training_loss, mean_validation_loss, loss_metric_name, plot_title_note, model.name, colab)


'''
Tuning a model using Grid Search.
Stores csv results files.

Parameters
    ----------
    df: the dataframe holding the data
    hyperparameters: a dictionary of hyperparameters to tune (pay attention to the dict keys - timesteps mandatory)
    cv_splits: number of splits to be used in the TimeSeries Cross Validator
    multisteps: number of multisteps
    scaler: a dictionary of scalers (must hold the scaler for the target feature, i.e., the 'value')
    build_model_function: the function to be used to build a deep learning model
    verbose: the verbose mode
    epochs: allows the user to set the number of epochs if he does not wants to tune it
    batch_size: allows the user to set the batch_size if he does not wants to tune it
    silent: log messages
    callbacks: list of callbacks to be used when training
    store_exp_ratio: when to save the results
    resume_at_experiment: to start at this experiment number, inclusive (a value of 3 starts at the third experiment). Starts at 1.
    seed: random seed to be used
    colab: writing to drive?    
'''


def tune_model(df, hyperparameters, cv_splits, multisteps, scalers, build_model_function, epochs=100, batch_size=32, verbose=0, silent=False, callbacks=[], store_exp_ratio=4, resume_at_experiment=-1, seed=91195003, colab=False):
    #obtain prepared param list and names
    param_list, param_names = prepare_param_list(hyperparameters, build_model_function)
    col_target_position = df.columns.get_loc('value')
    #initialize variables
    try:
        timesteps_position = param_names.index('timesteps')
    except:
        print("Error: You MUST tune the timesteps!")
        raise
    df_results = pd.DataFrame()
    i = 0
    total_experiments = len(param_list)
    if resume_at_experiment > total_experiments:
        print('********* Not that many experiments. Resuming at experiment ' + str(resume_at_experiment) + ' but only has ' + str(total_experiments) + '. *********')
        return
    model_name = ''
    #running the experiment
    for param_comb in param_list:
        #is to resume a previous experiment
        if resume_at_experiment > (i+1):
            i += 1
        else:
            param_names_aux = param_names.copy()
            #get the epochs if the user sets it. Otherwise, use default value (which can be passed as argument)
            if 'epochs' in param_names:
                epochs = param_comb[param_names.index('epochs')]
            else:
                param_names_aux.append('epochs')
                param_comb = param_comb + (epochs,)
            #get the batch_size if the user sets it. Otherwise, use default value (which can be passed as argument)
            if 'batch_size' in param_names:
                batch_size = param_comb[param_names.index('batch_size')]
            else:
                param_names_aux.append('batch_size')
                param_comb = param_comb + (batch_size,)
            #start experimenting
            start = time.time()
            i += 1
            if not silent:
                print('********* STARTING EXPERIMENT ' + str(i) + ' OUT OF ' + str(total_experiments) + ' *********')
                log_hyperparameters(param_names_aux, param_comb)
            #create the supervised problem
            X, y = tsp.to_supervised(df, param_comb[timesteps_position])
            #timeseries split for model validation
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            #storing loss values
            evaluate_loss = defaultdict(list)
            backtesting_loss = defaultdict(list)
            #model's validaty
            valid_model = True
            #cross-validate the results
            for train_index, test_index in tscv.split(X):
                #further split into training and validation sets
                train_idx, val_idx = tsp.split_dataset(train_index, perc=10)
                #build data
                X_train, y_train = X[train_idx], y[train_idx] 
                X_val, y_val = X[val_idx], y[val_idx] 
                X_test, y_test = X[test_index], y[test_index]
                #generically build model
                features = np.size(X_train, 2)  #the last dimension
                #model, model_pred = build_model(features, param_names_aux, param_comb, build_model_function)
                model = build_model(features, param_names_aux, param_comb, build_model_function, seed)
                model.summary()
                #if model is none it means it did not pass some condition
                if model is None:
                    valid_model = False
                    break
                else:
                    model_name = model.name
                    valid_model = True
                    #fit
                    if callbacks:
                        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, shuffle=False, verbose=verbose, callbacks=callbacks)
                    else:
                        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, shuffle=False, verbose=verbose)
                    # shap exp#
                    # ft = df.columns
                    # f = model.predict([X_test[:,i] for i in range(X_test.shape[1])]).flatten()
                    # explainer = shap.KernelExplainer(f, X.iloc[:50,:])
                    # shap_values = explainer.shap_values(X.iloc[299,:], nsamples=500)
                    # shap.initjs()
                    #shap.force_plot(explainer.expected_value, shap_values, ft)
                    # explainer = shap.DeepExplainer(model, X_train)
                    # shap_values = explainer.shap_values(X_test)
                    # shap.initjs()
                    # shap.force_plot(explainer.expected_value[0], shap_values[0][0], ft)
                    #evaluate model
                    metrics = model.evaluate(X_test, y_test, verbose=verbose)
                    for nr_metrics in range(len(model.metrics_names)):
                        evaluate_loss[model.metrics_names[nr_metrics]].append(metrics[nr_metrics])
                    #blind forecast for multisteps
                    blind_results = recursive_forecast(model, X_test.copy(), y_test, param_comb[timesteps_position], multisteps, features, scalers, col_target_position)
                    #storing backtesting results
                    backtesting_loss['RMSE_BLIND'].append(np.mean(blind_results[:, 2]))
                    backtesting_loss['MAE_BLIND'].append(np.mean(blind_results[:, 3]))
            #after the CV, i.e., finished this param_comb combination 
            run_time = time.time()-start
            if not silent and valid_model:
                print('Time the experiment took (s) = %.3f' %run_time)
                print('Mean backtesting [MAE, RMSE] values for BLIND forecasting are [%.3f, %.3f]' %(np.mean(backtesting_loss['MAE_BLIND']), np.mean(backtesting_loss['RMSE_BLIND'])))     
                for k, v in evaluate_loss.items():
                    print('Mean evaluation %s (normalized) value is %.3f' %(k, np.mean(v)))
                    denormalized_init = scalers['value'].inverse_transform([[-1]])
                    denormalized_diff = scalers['value'].inverse_transform([[-1+np.mean(v)]])
                    print('Mean evaluation %s (denormalized) value is %.3f' %(k, denormalized_diff-denormalized_init))                    
            #store data into dataframe
            df_results = store_experiment_results(df_results, param_names_aux, param_comb, backtesting_loss, evaluate_loss, run_time, i, store_exp_ratio, model_name, valid_model, scalers, colab)
            if valid_model == False and not silent:
                print('Did not run the experiment. Model skipped as it is not valid.')
            if not silent:
                print('********* FINISHED THE EXPERIMENT *********')


'''
Used to build the model passing the necessary hyperparameters the 
build model function.

It reads the build model signature (to obtain the parameters it is expecting)
and then finds which arguments it should set.

Parameters
    ----------
    features: number of features (to set the input shape)
    param_names: the hyperparameters' names
    param_combination: the combination of values
    seed: random seed to be used
        
Returns
    -------
    the model with the hyperparameters set
'''


def build_model(features, param_names, param_comb, build_model_function, seed):
    #find which hyperparameters are set in param_names
    #and exist in the build's mode signature
    buil_model_params = list(inspect.signature(build_model_function).parameters)
    #define which hyperparameters to set
    hyperparameter_to_be_set = list()
    for param in buil_model_params:
        if param in param_names:
            hyperparameter_to_be_set.append(param_names.index(param))
    #create mapping for function input
    mapping = dict()
    mapping['features'] = features
    mapping['seed'] = seed
    for hyperparameter in hyperparameter_to_be_set:
        mapping[param_names[hyperparameter]] = param_comb[hyperparameter]
    return build_model_function(**mapping)

def build_model_anomaly_detection(param_names, param_comb, build_model_function):
    #find which hyperparameters are set in param_names
    #and exist in the build's mode signature
    buil_model_params = list(inspect.signature(build_model_function).parameters)
    #define which hyperparameters to set
    hyperparameter_to_be_set = list()
    for param in buil_model_params:
        if param in param_names:
            hyperparameter_to_be_set.append(param_names.index(param))
    #create mapping for function input
    mapping = dict()
    for hyperparameter in hyperparameter_to_be_set:
        mapping[param_names[hyperparameter]] = param_comb[hyperparameter]
    return build_model_function(**mapping)


'''
Used to prepare the hyperparameter list.
Works as grid search. The cartesian product is used to obtain all hyperparameters
combination.

Parameters
    ----------
    hyperparameters: a dictionary of hyperparameter and the corresponding values 
        for tuning. Ex.:
            hyperparameters = {
                'h_layers': [4, 5],
                'h_neurons': [64, 128]
            }
        
Returns
    -------
    list of all combinations and the list with the hyperparameters names
'''


def prepare_param_list(hyperparameters, build_model_function):
    #find which hyperparameters are set in param_names
    #and exist in the build's mode signature
    build_model_params = list(inspect.signature(build_model_function).parameters)
    hyperparameters_names = list()
    hyperparameters_values = list()
    for item in hyperparameters.items():
        #only use the ones that are arguments of the build model function 
        if (item[0] in build_model_params) or (item[0] == 'epochs') or (item[0] == 'batch_size'):
            hyperparameters_names.append(item[0])
            hyperparameters_values.append(item[1])
    return list(itertools.product(*hyperparameters_values)), hyperparameters_names
     
def prepare_param_list_anomaly_detection(hyperparameters, build_model_function):
    #find which hyperparameters are set in param_names
    #and exist in the build's mode signature
    build_model_params = list(inspect.signature(build_model_function).parameters)
    hyperparameters_names = list()
    hyperparameters_values = list()
    for item in hyperparameters.items():
        #only use the ones that are arguments of the build model function 
        if (item[0] in build_model_params):
            hyperparameters_names.append(item[0])
            hyperparameters_values.append(item[1])
    return list(itertools.product(*hyperparameters_values)), hyperparameters_names

'''
Tuning a model using Grid Search.

Parameters
    ----------
    param_names: the hyperparameters' names
    param_combination: the combination of values
'''


def log_hyperparameters(param_names, param_combination):
    for i in range(len(param_combination)):
        print('%s = %s' %(param_names[i], param_combination[i]))
        
def hyperparameters_for_title(param_names, param_combination):
    title = ''
    for i in range(len(param_combination)):
        title += str(param_names[i]) + '=' + str(param_combination[i]) + ', '
    return title[:-2]

'''
Performing recursive forecasts for all the test data.
Calculates MAE and RMSE

Parameters
    ----------
    model: the trained model
    X_test: test inputs
    y_test: test labels
    timesteps: number of timesteps that make a sequence
    multisteps: number of timesteps to preduct
    features: number of features in the input sequence
    scaler: a dictionary with all the used scalers (must contain a scaler for the 'value' feature)
    col_target_position: location of the target feature (the 'value' one)
    evaluation_timesteps: the number of timesteps to evaluate
    
Returns
-------
    list of labels, predictions, rmse and mae scores
'''


def recursive_forecast(model, X_test, y_test, timesteps, multisteps, features, scaler, col_target_position, evaluation_timesteps=30):
    targer_scaler = scaler['value']    
    input_seq = X_test[-evaluation_timesteps:, :, :]
    labels_seq = y_test[-evaluation_timesteps:, :]
    #input_seq = X_test
    #labels_seq = y_test
    #iterate to get the results
    results = list()
    for i in range(len(input_seq)-(multisteps-1)):  #we cannot go through the entire input because we are multistep forecasting!
        inp = input_seq[i].copy()
        lab = labels_seq[i].copy()
        #for each step of the multistep
        labels = list()
        predictions = list()
        rmse_scores = list()
        mae_scores = list()
        for step in range(1, multisteps+1):
            #reshape
            inp = inp.reshape(1, timesteps, features)
            lab = lab.reshape(1, 1)
            #predict the value for the next timestep
            yhat = model.predict(inp, verbose=0)
            #invert normalized values
            lab_inversed = targer_scaler.inverse_transform(lab)
            yhat_inversed = targer_scaler.inverse_transform(yhat)
            #compute rmse and mae between true value and prediction
            mse_val = mean_squared_error(lab_inversed, yhat_inversed)
            mae_val = mean_absolute_error(lab_inversed, yhat_inversed)
            #store results
            labels.append(lab_inversed[0][0])
            predictions.append(yhat_inversed[0][0])
            rmse_scores.append(mse_val)
            mae_scores.append(mae_val)
            #insert a new value into the input sequence to predict the next timestep.
            if step != multisteps:
                #add yhat to the input sequence
                new_line = input_seq[i+step][-1].copy()
                np.put(new_line, col_target_position, yhat)
                new_line = new_line.reshape(1, features)
                inp = np.concatenate((inp[0], new_line))
                inp = inp[-timesteps:]
                #update label to the next timestep
                lab = labels_seq[i+step]
        results.append((np.array(labels), np.array(predictions), np.sqrt(np.mean(rmse_scores)), np.mean(mae_scores)))
    return np.array(results)

def recursive_forecast2(model, X_test, y_test, timesteps, multisteps, features, scaler, col_target_position, evaluation_timesteps=30):
    targer_scaler = scaler['value']    
    input_seq = X_test[-evaluation_timesteps:, :, :]
    labels_seq = y_test[-evaluation_timesteps:, :]
    results = list()
    for i in range(len(input_seq)-(multisteps-1)):  #we cannot go through the entire input because we are multistep forecasting!
        inp = input_seq[i].copy()
        lab = labels_seq[i].copy()
        #for each step of the multistep
        labels = list()
        predictions = []
        for step in range(1, multisteps+1):
            #reshape
            inp = inp.reshape(1, timesteps, features)
            lab = lab.reshape(1, 1)
            #predict the value for the next timestep
            yhat = model.predict(inp, verbose=0)
            #invert normalized values
            lab_inversed = targer_scaler.inverse_transform(lab)
            yhat_inversed = targer_scaler.inverse_transform(yhat)
            #store results
            labels.append(lab_inversed[0][0])
            predictions.append(yhat[0][0])
            #insert a new value into the input sequence to predict the next timestep.
            if step != multisteps:
                #add yhat to the input sequence
                new_line = input_seq[i+step][-1].copy()
                np.put(new_line, col_target_position, yhat)
                new_line = new_line.reshape(1, features)
                inp = np.concatenate((inp[0], new_line))
                inp = inp[-timesteps:]
                #update label to the next timestep
                lab = labels_seq[i+step]
    return np.array(predictions)

'''
Store the experiment results every 'store_exp_ratio' experiments.
Saves a csv file.

Parameters
    ----------
    df_results: dataframe where all results are stored
    param_names: the hyperparameters' names
    param_comb: the combination of values
    backtesting_loss: list with results from the backtesting evaluation
    evaluate_loss: list with results from the evaluation
    run_time: time the experiment took to complete
    iteration: current iteration (used in the file name as well)
    store_exp_ratio: when to save the results
    model_name: the model's name
    valid_model: is it a valid model to be stored?
    colab: writing to drive?
    
Returns
-------
    the updated dataframe with the results
'''


def store_experiment_results(df_results, param_names, param_comb, backtesting_loss, evaluate_loss, run_time, iteration, store_exp_ratio, model_name, valid_model, scaler, colab):
    #if not a valid model, store all results with -1
    #unless it is the first iteration (hence we do not know the shape of df_results and can't do anything)
    if valid_model == False:
        if len(df_results.columns) != 0:
            #copy the last row
            df_results = df_results.append(df_results.iloc[-1, :])
            df_results.iloc[-1, :] = -1
            df_results.iloc[-1, df_results.columns.get_loc('experiment')] = iteration
        else:
            return df_results
    else:
        #store data into dataframe
        #but first check if we must initialize it
        if len(df_results.columns) == 0:
            column_names = ['experiment']
            column_names = column_names + param_names + ['mae_blind', 'rmse_blind'] + [*evaluate_loss] + [*evaluate_loss]
            column_names.append('train_time')
            df_results = pd.DataFrame(columns=column_names)
        #add data to list for the dataframe
        new_row = [iteration] + list(param_comb)
        new_row.append(float(np.mean(backtesting_loss['MAE_BLIND'])))
        new_row.append(float(np.mean(backtesting_loss['RMSE_BLIND'])))
        for k, v in evaluate_loss.items():
            new_row.append(float(np.mean(v)))
        denormalized_init = scaler['value'].inverse_transform([[-1]])
        for k, v in evaluate_loss.items():
            denormalized_diff = scaler['value'].inverse_transform([[-1+float(np.mean(v))]])
            new_row.append(denormalized_diff-denormalized_init)            
        new_row.append(run_time)
        new_row_series = pd.Series(new_row, index=df_results.columns)
        #add new row to the end of the dataframe
        df_results = df_results.append(new_row_series, ignore_index=True)
    #store file every 'store_exp_ratio' experiments
    if iteration%store_exp_ratio == 0:
        model_name = model_name.replace(" ", "_")
        if not colab:
            from pathlib import Path
            Path('results').mkdir(parents=True, exist_ok=True)        
        #it is guaranteed STORE_MODEL_NAME has already been set
        filename = F'results/Experiments_' + model_name + '_' + time.strftime("%Y%m%d%H%M") + '_until_' + str(iteration) + '.csv'
        df_results.to_csv(filename, index=False)
    return df_results


def anomaly_detection_models(df, hyperparameters, cv_splits, scalers, build_model_function, epochs=100, batch_size=32, verbose=0, silent=False, store_exp_ratio=4, resume_at_experiment=-1, seed=91195003, colab=False):
    param_list, param_names = prepare_param_list_anomaly_detection(hyperparameters, build_model_function)
    i = 0
    total_experiments = len(param_list)
    if resume_at_experiment > total_experiments:
        print('********* Not that many experiments. Resuming at experiment ' + str(resume_at_experiment) + ' but only has ' + str(total_experiments) + '. *********')
        return
    
                
    df_results = pd.DataFrame()
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    if not silent:
        print('Train data:', str(train.shape) + '. Test data:', test.shape)
    
    if('timesteps' in param_names):
        for param_comb in param_list:
            #is to resume a previous experiment
            if resume_at_experiment > (i+1):
                i += 1
            else:
                evaluate_loss = defaultdict(list)
                backtesting_loss = defaultdict(list)
                temp_test = test.copy()
                param_names_aux = param_names.copy()
                if 'epochs' in param_names:
                    epochs = param_comb[param_names.index('epochs')]
                else:
                    param_names_aux.append('epochs')
                    param_comb = param_comb + (epochs,)
                #get the batch_size if the user sets it. Otherwise, use default value (which can be passed as argument)
                if 'batch_size' in param_names:
                    batch_size = param_comb[param_names.index('batch_size')]
                else:
                    param_names_aux.append('batch_size')
                    param_comb = param_comb + (batch_size,)
                i += 1
                temp_test = test.copy()
                if not silent:
                    print('********* STARTING EXPERIMENT ' + str(i) + ' OUT OF ' + str(total_experiments) + ' *********')
                    log_hyperparameters(param_names_aux, param_comb)
                    
                timesteps_position = param_names.index('timesteps')
                train_X, train_Y = tsp.to_supervised(train, param_comb[timesteps_position])
                test_X, test_Y = tsp.to_supervised(test, param_comb[timesteps_position])
                #timeseries split for model validation
                tscv = TimeSeriesSplit(n_splits=cv_splits)
                #cross-validate the results
                for train_index, val_index in tscv.split(train_X):
                    #build data
                    X_train, y_train = train_X[train_index], train_Y[train_index] 
                    X_val, y_val = train_X[val_index], train_Y[val_index] 
                    #generically build model
                    features = np.size(X_train, 2)  #the last dimension
                    model = build_model(features, param_names_aux, param_comb, build_model_function, seed)
                    model.summary()
                    model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=epochs, batch_size=batch_size, shuffle=False, verbose=verbose)
               
                test_X_pred = model.predict(test_X)
                test_mae_loss = np.mean(np.abs(test_X_pred - test_X), axis=1)
                
                temp_test = temp_test.iloc[param_comb[timesteps_position]:].copy()
                temp_test['ph'] = scalers['value'].inverse_transform([temp_test['value']])[0]
                if features > 1:
                    test_mae_loss = test_mae_loss.mean(axis=1)
                temp_test['loss'] = test_mae_loss
                temp_test['threshold'] = np.mean(test_mae_loss) + 3 * np.std(test_mae_loss)
                temp_test['anomaly'] = temp_test.loss > temp_test.threshold
                
                df_results = store_experiment_results(df_results, param_names_aux, param_comb, backtesting_loss, evaluate_loss, np.mean(temp_test['loss']), i, store_exp_ratio, model.name, True, scalers, colab)
                
                anomaly_perc = len(temp_test[temp_test['anomaly']==True])/len(temp_test)*100
                if not silent:
                    print('Anomaly percentage', str(anomaly_perc) + '%')
                if(anomaly_perc < 25 and anomaly_perc > 0):
                    plots.line_threshold(temp_test.loss, temp_test.threshold)
                    plots.line_scatter(temp_test.ph, temp_test[temp_test['anomaly'] == True].index, temp_test[temp_test['anomaly'] == True].ph, hyperparameters_for_title(param_names_aux, param_comb))
                if not silent:
                    print('********* FINISHED THE EXPERIMENT *********')
    else:
        for param_comb in param_list:
            #is to resume a previous experiment
            if resume_at_experiment > (i+1):
                i += 1
            else:
                param_names_aux = param_names.copy()
                i += 1
                temp_test = test.copy()
                if not silent:
                    print('********* STARTING EXPERIMENT ' + str(i) + ' OUT OF ' + str(total_experiments) + ' *********')
                    log_hyperparameters(param_names_aux, param_comb)
                model = build_model_anomaly_detection(param_names_aux, param_comb, build_model_function)
                model.fit(train)
                temp_test['anomaly'] = model.predict(temp_test)
                print('FASE DE TREINO ------------')
                print('FIM DADOS TREINO ---------------')
                anomaly_perc = len(temp_test[temp_test['anomaly']==-1])/len(temp_test)*100
                if not silent:
                    print('Anomaly percentage', str(anomaly_perc) + '%')
                if(anomaly_perc < 25 and anomaly_perc > 0):
                    plots.line_scatter(temp_test['value'], temp_test[temp_test['anomaly']==-1].index, temp_test[temp_test['anomaly']==-1].value, hyperparameters_for_title(param_names_aux, param_comb))
                if not silent:
                    print('********* FINISHED THE EXPERIMENT *********\n')
    return model