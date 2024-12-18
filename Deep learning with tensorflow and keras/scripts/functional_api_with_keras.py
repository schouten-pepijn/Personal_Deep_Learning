import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils

#%% CREATE A MODEL WITH FUNCTIONAL API
params = {"shape": 28, "activation": "relu", "classes": 10,
          "units": 12, "optimizer": "adam", "epochs": 1,
          "kernel_size": 3, "pool_size": 2, "dropout": 0.5}


inputs = keras.Input(shape=(params["shape"], params["shape"], 1))

conv2D = layers.Conv2D(32, kernel_size=(params["kernel_size"],
                                        params["kernel_size"]),
                       activation=params["activation"])(inputs)

maxPooling2D = layers.MaxPooling2D(pool_size=(params["pool_size"],
                                              params["pool_size"]))(conv2D)

conv2D_2 = layers.Conv2D(32, kernel_size=(params["kernel_size"],
                                          params["kernel_size"]),
                         activation=params["activation"])(maxPooling2D)

maxPooling2D_2 = layers.MaxPooling2D(pool_size=(params["pool_size"],
                                                params["pool_size"]))(conv2D_2)

flatten = layers.Flatten()(maxPooling2D_2)

dropout = layers.Dropout(params["dropout"])(flatten)

outputs = layers.Dense(params["classes"], activation="softmax")(dropout)

model = keras.Model(inputs=inputs, outputs=outputs,
                    name="mnist_model")

# utils.plot_model(model, "functional_api_model.png", show_shapes=True)
print(model.summary())


#%% CONVERT FUNCTIONAL MODEL TO SEQUENTIAL MODEL
seq_model = keras.models.Sequential()
for layer in model.layers:
    seq_model.add(layer)

print(seq_model.summary())


#%% shared input layer model
inputs = keras.Input(shape=(10,))
dense_shared = layers.Dense(50, activation="relu")(inputs)

dense_1 = layers.Dense(50, activation="relu")(dense_shared)
dense_2 = layers.Dense(50, activation="relu")(dense_shared)

merged_layers = layers.concatenate([dense_1, dense_2])
outputs = layers.Dense(10, activation="relu")(merged_layers)

shared_model = keras.Model(inputs=inputs, outputs=outputs, name="shared_layer")

# utils.plot_model(shared_model, "shared_layer_model.png", show_shapes=True)
print(shared_model.summary())


#%% MULTIPLE INPUT MODELS
input_1 = keras.Input(shape=(16,))
x_1 = layers.Dense(8, activation="relu")(input_1)

input_2 = keras.Input(shape=(32,))
x_2 = layers.Dense(8, activation="relu")(input_2)

added_layers = layers.add([x_1, x_2])
outputs = layers.Dense(4)(added_layers)
input_model = keras.Model(inputs=[input_1, input_2], outputs=outputs)

print(input_model.summary())
# utils.plot_model(input_model, "multiple_input_model.png", show_shapes=True)


#%% MULTIPLE OUTPUT MODEL
inputs = keras.Input(shape=(16,))
x = layers.Dense(8, activation="relu")(inputs)
output_1 = layers.Dense(3, activation="relu")(x)
output_2 = layers.Dense(3, activation="relu")(x)

output_model = keras.Model(inputs=inputs, outputs=[output_1, output_2])

print(output_model.summary())
utils.plot_model(output_model, "multiple_output_model.png", show_shapes=True)