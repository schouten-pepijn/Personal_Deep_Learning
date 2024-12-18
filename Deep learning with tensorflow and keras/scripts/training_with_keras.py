import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
import tensorflow.keras as keras

from matplotlib import pyplot as plt

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score

#%% DATA LOADING
csv_path = '/Users/pepijnschouten/Desktop/Deep learning with tensorflow and keras/insurance.csv'
df = pd.read_csv(csv_path)


#%% PREPROCESSING
print(df.head())
print(df.info())
# age -> int | sex -> object | bmi -> float | children -> int | smoker -> object
# region -> object | charges -> float

print(df.describe())

for key in df.keys():
    print(f'{key}: {pd.unique(df[key])}')
    
# convert smoker and sex to numerical labels
label_encoder = LabelEncoder()

df['smoker'] = label_encoder.fit_transform(df['smoker'])
df['sex'] = label_encoder.fit_transform(df['sex'])

print(pd.unique(df['smoker']))
print(pd.unique(df['sex']))

# cast ints to flaots
ints = df.select_dtypes(include=['int64']).columns.tolist()
df = df.astype({key:'float' for key in ints})

print(df.info())

# split features and labels
X = df.drop('charges', axis=1)
y = df['charges']

# one hot encode the region column and scale to 0 - 1
transformer = ColumnTransformer(transformers=[('region', OneHotEncoder(handle_unknown='ignore',
                                                                       drop='first'), ['region'])],
                                remainder=MinMaxScaler())
X = transformer.fit_transform(X)

print(np.min(X, axis=1), np.max(X, axis=1))

# create train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%% CREATE A KERAS MODEL
model = keras.Sequential()
model.add(keras.layers.Input(shape=(X_train.shape[1],)))
model.add(keras.layers.Dense(600, activation='relu',
                             kernel_initializer='glorot_uniform',
                             name='fc1'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(1000, activation='relu',
                             kernel_initializer='glorot_uniform',
                             name='fc2'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(600, activation='relu',
                             kernel_initializer='glorot_uniform',
                             name='fc3'))
model.add(keras.layers.Dense(1, activation=None,
                             name='fc4'))

print(model.summary())


#%% COMPILE THE MODEL
optimizer = keras.optimizers.Adam()
criterion = keras.losses.MeanSquaredError()
metrics = keras.metrics.RootMeanSquaredError()
model.compile(optimizer=optimizer, loss=criterion, metrics=[metrics])


#%% TRAIN THE MODEL
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
             keras.callbacks.ModelCheckpoint(
                 filepath='model_checkpoint.weights.h5',
                 save_weights_only=True,
                 monitor='val_root_mean_squared_error',
                 mode='max',
                 save_best_only=True)]

device = '/CPU:0'
with tf.device(device):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                        batch_size=32, epochs=200, callbacks=callbacks)


#%% EVALUATE THE MODEL
metrics_df = pd.DataFrame(history.history)
metrics_df[["loss", "val_loss"]].plot()
plt.show()

metrics_df[["root_mean_squared_error", "val_root_mean_squared_error"]].plot()
plt.show()


#%% LOAD THE MODEL
model.load_weights('model_checkpoint.weights.h5')

#%% MAKE PREDICTIONS
y_pred = model.predict(X_test)


#%% MAKE A CROSS VALIDATION SCORE
def create_model():
    """Create a feedforward neural network model for regression."""
    model = keras.Sequential([
        keras.layers.Dense(600, activation='relu', input_shape=(X_train.shape[1],),
                           kernel_initializer='glorot_uniform', name='dense_1'),
        keras.layers.BatchNormalization(name='batch_norm_1'),
        keras.layers.Dropout(rate=0.3, name='dropout_1'),
        keras.layers.Dense(1000, activation='relu', kernel_initializer='glorot_uniform',
                           name='dense_2'),
        keras.layers.BatchNormalization(name='batch_norm_2'),
        keras.layers.Dense(600, activation='relu', kernel_initializer='glorot_uniform',
                           name='dense_3'),
        keras.layers.Dense(1, activation=None, name='dense_4')
    ])
    return model

cvs_model = KerasRegressor(model=create_model, batch_size=32, 
                           optimizer='adam', metrics=['root_mean_squared_error'],
                           loss='mean_squared_error',
                           validation_split=0.2, epochs=10)

rmses = cross_val_score(estimator=cvs_model, X=X_train, y=y_train,
                        cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
