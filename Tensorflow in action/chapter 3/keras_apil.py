import requests
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
import tensorflow.keras.backend as K


#%% DATA PREPARATIOM
path = "/Users/pepijnschouten/Desktop/Tensorflow in action/chapter 3/iris.data"

iris_df = pd.read_csv(path, header=None)
print(iris_df.sample(5))

iris_df.columns = ["sepal_length",
                   "sepal_width",
                   "petal_width",
                   "petal_length",
                   "labels"]

le = LabelEncoder()

iris_df["labels"] = le.fit_transform(iris_df["labels"])

iris_df = iris_df.sample(frac=1.0, random_state=87)
x = iris_df.drop("labels", axis=1)

ss = StandardScaler()
x = ss.fit_transform(x)

y = pd.get_dummies(iris_df["labels"], dtype=int)

#%% SEQUENTIAL MODEL
K.clear_session()
model = Sequential([
    Input(shape=(4,)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(3, activation="softmax")])

print(model.summary())

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

history = model.fit(x, y,
                    batch_size=32,
                    epochs=25)

history = pd.DataFrame(history.history)
print(history.iloc[-1])

#%% FUNCTIONAL API MODEL
inp_1 = Input(shape=(4,))
inp_2 = Input(shape=(2,))

out_1 = Dense(16, activation="relu")(inp_1)
out_2 = Dense(16, activation="relu")(inp_2)
out = Concatenate(axis=1)([out_1, out_2])
out = Dense(16, activation="relu")(out)
out = Dense(3, activation="softmax")(out)

model = Model(inputs=[inp_1, inp_2], outputs=out)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

print(model.summary())
keras.utils.plot_model(model)

pca_model = PCA(n_components=2, random_state=87)
x_pca = pca_model.fit_transform(x)

history = model.fit([x, x_pca], y,
                    batch_size=32,
                    epochs=25)

history = pd.DataFrame(history.history)
print(history.iloc[-1])


#%% SUBCLASSING API MODEL
class MulBiasDense(layers.Layer):
    def __init__(self, units=32, input_dim=12,
                 activation=None):
        super().__init__()
        self.units = units
        self.activation = activation
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],
                   self.units),
            initializer="glorot_uniform",
            trainable=True)
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="glorot_uniform",
            trainable=True)
        self.b_mul = self.add_weight(
            shape=(self.units,),
            initializer="glorot_uniform",
            trainable=True)
    
    def call(self, inputs):
        out = (tf.matmul(inputs, self.w) + self.b) * self.b_mul
        return layers.Activation(self.activation)(out)

K.clear_session()

inp = Input(shape=(4,))
out = MulBiasDense(units=32, activation="relu")(inp)
out = MulBiasDense(units=16, activation="relu")(out)
out = Dense(3, activation="softmax")(out)

model = Model(inputs=inp, outputs=out)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

history = model.fit(x, y,
                    batch_size=32,
                    epochs=25)

history = pd.DataFrame(history.history)
print(history.iloc[-1])
    

# page 74 of text


