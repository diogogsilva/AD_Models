# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:33:14 2020

@author: bruno
"""

import matplotlib.pyplot as plt
from pathlib import Path
import time

'''
Plotting learning curves per time series cross validator split.

Parameters
    ----------
    data: list with hystory objects (one for each time series split)
    loss_metric_name: the name of the loss metric being used
    title_note: indentify if it is the lighter or heavier combination of hyperparameters
    colab: writing to drive?    
''' 
def plot_learning_curve_per_split(data, loss_metric_name, title_note, model_name, colab):
    cv_splits = len(data)
    plt.figure(figsize=(12,8))     
    for hist, i in zip(data, range(cv_splits)):  
        plt.subplot(cv_splits, 1, i+1)   
        if i == 0:
            plt.title('Training Loss vs Validation Loss (per Training Split) - ' + title_note.replace("_", " ") + ' - ' + model_name.replace("_", " "))
        plt.plot(range(1, len(hist.epoch)+1), hist.history['loss'], label='train')
        plt.plot(range(1, len(hist.epoch)+1), hist.history['val_loss'], label='val')
        plt.ylim([min(min(hist.history['loss']), min(hist.history['val_loss']))/2, max(max(hist.history['loss']), max(hist.history['val_loss']))+0.2])
        plt.legend(['Training split ' + str(i+1) + ' - train loss', 'Training split ' + str(i+1) + ' - val loss'], loc='upper right')
        plt.xlim([1, len(hist.epoch)])  
        #plt.xticks(range(1, max(hist.epoch)+2)) #problematic when huge amount of epochs
    plt.ylabel('Training ' + loss_metric_name + ' (Normalized)')   
    plt.xlabel('Epochs')
    #store image
    model_name.replace(" ", "_")
    if not colab:
        plt.show() #show() breaks colab
        Path('results').mkdir(parents=True, exist_ok=True)        
    filename = F'results/LearningCurve_' + model_name + '_' + time.strftime("%Y%m%d%H%M") + '_Difficulty_' + title_note + '_Epochs_' + str(len(hist.epoch)) + '.pdf'
    plt.savefig(filename, bbox_inches='tight', format='pdf')
    

'''
Plotting aggregated learning curves

Parameters
    ----------
    mean_training_loss: mean training loss (for the cv split) for each epoch
    mean_validation_loss: mean validation loss (for the cv split) for each epoch
    loss_metric_name: the name of the loss metric being used
    title_note: indentify if it is the lighter or heavier combination of hyperparameters
    colab: writing to drive?    
''' 
def plot_learning_curve_aggregated_splits(mean_training_loss, mean_validation_loss, loss_metric_name, title_note, model_name, colab):
    plt.figure(figsize=(12,4))     
    plt.title('Training Loss vs Validation Loss (Aggregated Splits) - ' + title_note.replace("_", " ") + ' - ' + model_name.replace("_", " "))
    plt.ylabel('Training ' + loss_metric_name + ' (Normalized)')
    plt.plot(range(1, len(mean_training_loss)+1), mean_training_loss, label='train')
    plt.plot(range(1, len(mean_validation_loss)+1), mean_validation_loss, label='val')
    plt.ylim([min(min(mean_training_loss), min(mean_validation_loss))/2, max(max(mean_training_loss), max(mean_validation_loss))+0.2])
    plt.legend(loc='upper right')
    plt.xlim([1, len(mean_training_loss)])  
    plt.xlabel('Epochs')
    #store image
    model_name = model_name.replace(" ", "_")
    if not colab:
        plt.show() #show() breaks colab
        Path('results').mkdir(parents=True, exist_ok=True)        
    filename = F'results/LearningCurve_' + model_name + '_' + time.strftime("%Y%m%d%H%M") + '_Difficulty_' + title_note + '_Epochs_' + str(len(mean_training_loss)) + '.pdf'
    plt.savefig(filename, bbox_inches='tight', format='pdf')
    
    
def line_scatter(line, scatter_x, scatter_y, title=''):
    plt.rcParams["figure.figsize"] = (20,9)
    fig, ax1 = plt.subplots()  
    ax1.plot(line, label='ph')
    ax1.scatter(scatter_x, scatter_y, c='red', label="anomaly")
    plt.title(title)
    plt.show()
    
def line_threshold(loss, threshold):
    loss.plot(kind='line')
    threshold.plot(kind='line')
    plt.show()