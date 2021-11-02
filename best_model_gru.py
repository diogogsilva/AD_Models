import pandas as pd
import load_dataset as ld
import timeseries_preparation as tsp
import experiments as exp
import build_model as bm
import main_AD as ad_algorithms
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

COLAB = False
SEED = 91195003

df = pd.read_csv(r'results/Experiments_gru_model.csv', index_col=['experiment'])
df['mean_loss'] = (df['mae_blind'] + df['rmse_blind'] + df['loss'] + df['mae'] + df['rmse'])/5
#### df['mae_blind']
print(df.sort_values(['mean_loss']))

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
    'layers': [2],
    'neurons': [64],
    'dropout_rate': [0.5],
    'activation': ['tanh'],
    'timesteps': [7],
    'batch_size': [30]
}
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=25, min_lr=0.00005),
             tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, mode='min', verbose=0, patience=40)]

param_list, param_names = exp.prepare_param_list(hyperparameters, bm.build_gru_model)
model = exp.build_model(1, param_names, param_list[0], bm.build_gru_model, SEED)
print(model.summary())

X, y = tsp.to_supervised(df_data, param_list[0][param_names.index('timesteps')])
tscv = TimeSeriesSplit(n_splits=cv_splits)
for train_index, test_index in tscv.split(X):
  train_idx, val_idx = tsp.split_dataset(train_index, perc=10)
  X_train, y_train = X[train_idx], y[train_idx] 
  X_val, y_val = X[val_idx], y[val_idx] 
  X_test, y_test = X[test_index], y[test_index]
  if callbacks:
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=param_list[0][param_names.index('batch_size')], shuffle=False, verbose=1, callbacks=callbacks)

#generated_data = np.random.uniform(low=6, high=9, size=9)
generated_data = [7.61, 7.075, 7.525, 6.9849, 7.105, 7.185, 7.04, 7.15, 7.165]
generated_df = pd.DataFrame(generated_data, columns=['value'])
generated_df['value'] = scalers['value'].transform(generated_df[['value']])
X_generated, y_generated = tsp.to_supervised(generated_df , param_list[0][param_names.index('timesteps')])
predictions = exp.recursive_forecast2(model, X_generated, y_generated, param_list[0][param_names.index('timesteps')], 2, 1, scalers, generated_df.columns.get_loc('value'))
for yhat in predictions:
    generated_df = generated_df.append({'value':yhat}, ignore_index=True)

if_model = ad_algorithms.get_best_AD_model('if')
ocsvm_model = ad_algorithms.get_best_AD_model('ocsvm')
print(if_model.predict(generated_df))
print(ocsvm_model.predict(generated_df))
generated_df['ph'] = scalers['value'].inverse_transform(generated_df[['value']])
generated_df['ph'].plot(kind='line')