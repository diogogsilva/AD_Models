# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:44:35 2020

@author: bruno
"""

import pandas as pd

'''
Available Datasets
'''
TRAFFIC_FLOW_UNI_MASKED_DATASET = r'datasets/TrafficFlow_BRAGA_N14_Masked_Univariate_072018_062020.csv'
TRAFFIC_FLOW_MULTI_MASKED_DATASET = r'datasets/TrafficFlow_BRAGA_N14_Masked_Multivariate_072018_062020.csv'
TRAFFIC_FLOW_UNI_DATASET = r'datasets/TrafficFlow_BRAGA_N14_Univariate_072018_062020.csv'
TRAFFIC_FLOW_MULTI_DATASET = r'datasets/TrafficFlow_BRAGA_N14_Multivariate_072018_062020.csv'
ULTRAVIOLET_UNI_DATASET = r'datasets/UV_BRAGA_Univariate_012018_062020.csv'
ULTRAVIOLET_MULTI_DATASET = r'datasets/UV_BRAGA_Multivariate_012018_062020.csv'
ENERGY_UNI_DATASET = r'datasets/univariate_2nd_approach.csv'
ENERGY_MULTI_DATASET = r'energy_multivariate_v4.csv'
PH_MULTI_DATASET = r'ph_entrada_guimaraes.csv'
PH_UNI_DATASET = r'datasets/ph_v2.csv'
AMONIA_MULTI_DATASET = r'datasets/amonia_multivariate.csv'

'''
Load Traffic Flow Dataset

Parameters
    ----------
    univariate: if True load univariate dataset. Otherwise, multivariate
    mask: if True load the masked version of the dataset
    colab : if True is to run in colab. Otherwise, locally
        
Returns
    -------
    dataset as dataframe
'''


def load_traffic_dataset(univariate=True, mask=False, colab=False):
    if univariate:
        if mask:
            return load_dataset(TRAFFIC_FLOW_UNI_MASKED_DATASET, colab)
        else:
            return load_dataset(TRAFFIC_FLOW_UNI_DATASET, colab)
    else:    
        if mask:
            return load_dataset(TRAFFIC_FLOW_MULTI_MASKED_DATASET, colab)
        else:
            return load_dataset(TRAFFIC_FLOW_MULTI_DATASET, colab)
    

'''
Load Ultraviolet Index Dataset

Parameters
    ----------
    univariate: if True load univariate dataset. Otherwise, multivariate
    colab : if True is to run in colab. Otherwise, locally
        
Returns
    -------
    dataset as dataframe
'''


def load_uv_dataset(univariate=True, colab=False):
    if univariate:
        return load_dataset(ULTRAVIOLET_UNI_DATASET, colab)
    else:    
        return load_dataset(ULTRAVIOLET_MULTI_DATASET, colab)


def load_energy_dataset(univariate=True, colab=False):
    if univariate:
        return load_dataset(ENERGY_UNI_DATASET, colab)
    else:    
        return load_dataset(ENERGY_MULTI_DATASET, colab)


def load_ph_dataset(univariate=True, colab=False):
    if univariate:
        return load_dataset(PH_UNI_DATASET, colab)
    else:
        pass

def load_amonia_dataset(univariate=False, colab=False):
    if univariate:
        pass
    else:
        return load_dataset(AMONIA_MULTI_DATASET, colab)


'''
Load a dataset

Parameters
    ----------
    dataset: the dataset's path
    colab : if True is to run in colab. Otherwise, locally
        
Returns
    -------
    dataset as dataframe
'''


def load_dataset(dataset, colab=False):
    df = pd.read_csv(dataset, infer_datetime_format=True, parse_dates=['timestep'], index_col=['timestep'])
    return df

# =============================================================================
# 	#Not required. Datasets must be placed in GDrive.
#     if colab:
#         #give permission to save to drive
#         save_to_drive()
#         #ask the user for the dataset
#         from google.colab import files
#         import io
#         uploaded = files.upload()
#         for fn in uploaded.keys():
#             print('Uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
#             df = pd.read_csv(io.BytesIO(uploaded[fn]), infer_datetime_format=True, parse_dates=['timestep'], index_col=['timestep'])    
#         return df
#     else:
#    	  df = pd.read_csv(dataset, infer_datetime_format=True, parse_dates=['timestep'], index_col=['timestep'])
#    	  return df
# =============================================================================


'''
Get permission to save a file to Google Drive account
'''

#def save_to_drive():
#    from google.colab import drive
#    drive.mount('/content/gdrive')