import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import layers, models

from jax import jit
import jax.numpy as jnp
import jax.random as random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% DATA IMPORT
(x_train, y_train), (x_test, y_test) = load_data()

x_train, x_test = jnp.array(x_train), jnp.array(x_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

print(f"x_train has shape: {x_train.shape}")
print(f"x_test has shape: {x_test.shape}")


#%% DATA PREPROCESSING
norm_x_train = jnp.reshape((x_train - 128.0) / 128.0, (-1,784))
norm_x_test = jnp.reshape((x_test - 128.0) / 128.0, (-1,784))

@jit
def gen_masked_inputs(x, p, seed=42):
    key = random.key(seed)
    mask = random.binomial(key, n=1, p=p, shape=x.shape, dtype='float32')
    out = jnp.where(x > 0, mask * x, 0)
    return out

masked_x_train = gen_masked_inputs(norm_x_train, p=0.5)


#%% DATA INSPECTION
fig, ax = plt.subplots(3, 3, tight_layout=True)
for i, a in enumerate(ax.flat):
    a.imshow(masked_x_train[i].reshape(28, 28), cmap='grey')
    a.axis('off')
plt.show()


#%% CREATE MODEL
ac = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(784, activation='tanh')
    ])

criterion = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam()
ac.compile(loss=criterion, optimizer=optimizer)


#%% TRAIN MODEL
epochs = 15
batch_size = 64
history = ac.fit(masked_x_train, norm_x_train,
                 batch_size=batch_size, epochs=epochs)


#%% MODEL EVALUATION
history_df = pd.DataFrame(history.history)

plt.figure()
plt.plot(history_df['loss'])
plt.show()


#%% MODEL PREDICTION
norm_x_train_sample = norm_x_train[:6]

norm_masked_x_train_sample = gen_masked_inputs(norm_x_train_sample, p=0.5)

y_pred = ac.predict(norm_masked_x_train_sample)

fig, ax = plt.subplots(2, 6, tight_layout=True)
for i, a in enumerate(ax.flat):
    if i < 6:
        a.imshow(norm_masked_x_train_sample[i].reshape(28,28), cmap='grey')
    else:
        a.imshow(y_pred[i-6].reshape(28, 28), cmap='grey')
    a.axis('off')
plt.show()