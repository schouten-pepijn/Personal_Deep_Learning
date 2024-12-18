import pandas as pd

import re

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils

import numpy as np

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# nltk.download("stopwords")


#%% DATA IMPORT
data_path = "/Users/pepijnschouten/Desktop/Deep learning with tensorflow and keras/Consumer_Complaints.csv"
df = pd.read_csv(data_path)

print(df.columns)
print(df.info())

df = df[["Consumer complaint narrative", "Product"]]
df = df.dropna()

print(df.sample(5))


#%% DATA CLEANING
symbols_regex = re.compile('[/(){}\[\][@,;]]')
bad_symbols_regex = re.compile('[^0-9a-z #+_)]')

def clean_text(text):
    text = text.replace('\d+','')
    text = text.lower()
    text = symbols_regex.sub(' ', text)
    text = bad_symbols_regex.sub('', text)
    text = text.replace('x', '')
    return text

df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(clean_text)


#%% LABEL EXPLORATION
print(df['Product'].value_counts().sort_values(ascending=False))
print(len(pd.unique(df['Product'])))


#%% TEXT VECTORIZATION
vectorize_layer = keras.layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=5000,
    output_mode='int',
    output_sequence_length=512)

vectorize_layer.adapt(df['Consumer complaint narrative'], batch_size=None)

x_padded = vectorize_layer(df['Consumer complaint narrative'])
x_padded = x_padded.numpy()

print(x_padded[:5])

le = LabelEncoder()
y = le.fit_transform(df['Product'])
y = utils.to_categorical(y, num_classes=len(pd.unique(df['Product'])))


x_train, x_test, y_train, y_test = train_test_split(x_padded, y, test_size=0.3,
                                                    random_state=42)


#%% KERAS MODEL
classifier = models.Sequential()
classifier.add(layers.Embedding(50000, 100))
classifier.add(layers.SpatialDropout1D(0.2))
classifier.add(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
classifier.add(layers.Dense(18, activation='softmax'))


optimizer = keras.optimizers.Adam()
criterion = keras.losses.CategoricalCrossentropy()
metrics = keras.metrics.Accuracy()
classifier.compile(optimizer=optimizer,
                   loss=criterion,
                   metrics=[metrics])

with tf.device('/CPU:0'):
    classifier.fit(x_train, y_train, epochs=100, batch_size=64)








