import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


#%% DATA

data_path = "/Users/pepijnschouten/Desktop/Deep learning with tensorflow and keras/btc_4h_data_2018_to_2024-09-06.csv"
df = pd.read_csv(data_path)

print(df.columns)
print(df.info())

df = df[['Open time', 'Close']]
df['Open time'] = pd.DatetimeIndex(df['Open time'])
df = df.groupby(['Open time']).sum().reset_index()
df = df.sort_values(by=['Open time'])
df = df.set_index('Open time')


#%% PREPROCESSING

scaler = MinMaxScaler(feature_range=(0,1))
scaled_df = scaler.fit_transform(df)

# create sequences of 60 intervals to predict the next interval
seq_length = 60
x_train, y_train = [], []
for i in range(seq_length, len(scaled_df)):
    x_train.append(scaled_df[i-seq_length:i, 0])
    y_train.append(scaled_df[i, 0])
    
# convert to 3D
y_train = np.array(y_train).astype('float32')
x_train = np.array(x_train).astype('float32')
x_train = x_train.reshape((np.shape(x_train)[0], np.shape(x_train)[1], 1))


#%% KERAS MODEL

regressor = Sequential([
    Input((np.shape(x_train)[1], 1)),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    Dense(units=1),
    Dropout(0.2),
    Dense(units=1)
    ])

regressor.summary()


lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3*10**(-epoch/20))
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
regressor.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])

early_stopping = keras.callbacks.EarlyStopping(monitor='loss', mode='min',
                                               patience=20)
mc = keras.callbacks.ModelCheckpoint('best_lstm_model.weights.h5', monitor='loss',
                                     mode='min', verbose=0, save_best_only=True,
                                     save_weights_only=True)

epochs = 100
device = '/GPU:0'
with tf.device(device):
    history = regressor.fit(x_train, y_train, epochs=epochs, batch_size=32, 
                            callbacks=[mc, lr_schedule, early_stopping])


#%%
plt.figure()
plt.plot(history.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
