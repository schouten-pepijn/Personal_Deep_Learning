import datetime
import numpy as np
import os

from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as utils
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Input
# from tensorflow.keras.utils import np_utils


#%% DATA IMPORTS
iris = datasets.load_iris()
x, y = iris.data, iris.target

x = normalize(x, axis=0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    stratify=y, random_state=42)

y_train, y_test = utils.to_categorical(y_train), utils.to_categorical(y_test)


#%% CREATE MODEL
def create_model():
    model = Sequential()
    model.add(Input((4,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

model = create_model()

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

#%% MODEL TRAINING
def train_model():
    model = create_model()
    
    tensorboard_callback = callbacks.TensorBoard(logdir,
                                                 histogram_freq=1,
                                                 write_graph=True,
                                                 write_images=True)
    
    model.fit(x=x_train,
              y=y_train,
              epochs=50,
              validation_data=(x_test,
                               y_test),
              callbacks=[tensorboard_callback])
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    
    print(f"Test Accuracy: {test_acc} \nTest Loss: {test_loss}")
    
tf.debugging.experimental.enable_dump_debug_info(
    "/tmp/tfdb2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)

train_model()
    