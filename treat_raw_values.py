# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:53:13 2021

@author: diogo.s.a
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta


def get_data(indicator):
    data = pd.read_csv('datasets/' + indicator + '.csv', parse_dates=True)
    data.timestep = pd.to_datetime(data.timestep)
    data.rename(columns={'value': indicator}, inplace=True)
    data['timestep'] = data.timestep - data.timestep.dt.weekday * np.timedelta64(1, 'D')
    data = data.groupby('timestep').agg('mean')
    return data


data_amonia = get_data('amonia')
data_azoto = get_data('azoto_total')

print(pd.date_range(data_amonia.index[0], data_amonia.index[-1], freq='W-MON').difference(data_amonia.index))
print(pd.date_range(data_amonia.index[0], data_amonia.index[-1], freq='W-MON').difference(data_azoto.index))
data = pd.merge(data_amonia, data_azoto, left_index=True, right_index=True, how="left")
print(data[data.isna().any(axis=1)])
while len(data[data.isna().any(axis=1)]) > 0:
    data.azoto_total = data.azoto_total.fillna(data.azoto_total.rolling(3).mean().shift())
print(data.shape)
print(data.corr(method='spearman'))
data.to_csv('datasets/amonia_multivariate.csv')











''' # CSV WITH VARIOUS INDICATORS TO MULTIPLE CSV'S

dataset = pd.read_csv('datasets/efluente_tratado_original.csv')
dataset.drop(columns=['indicator_type', 'units', 'sub_type', 'input', 'city_name'], inplace=True)
dataset = dataset[dataset['indicator_name'] == 'amonia']
print(dataset.shape)
dataset.drop(columns=['indicator_name'], inplace=True)
dataset = dataset.rename(columns={'date':'timestep'})
dataset.set_index('timestep', inplace=True)
print(dataset.shape)
dataset.to_csv('datasets/amonia.csv')

'''
