import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

folder_dir = "/Users/pepijnschouten/Desktop/Tensorflow in action/chapter 3/Flower_color"
data_dir = os.path.join(folder_dir, "flower_images", "flower_images") + os.path.sep
csv_dir = os.path.join(data_dir, "flower_labels.csv")

#%% USING TF.DATA
csv_ds = tf.data.experimental.CsvDataset(filenames=csv_dir,
                                         record_defaults=("",-1),
                                         header=True)


for i, item in enumerate(csv_ds.as_numpy_iterator()):
    print(item)
    if i == 4:
        break
    
fnames_ds = csv_ds.map(lambda a,b: a)
labels_ds = csv_ds.map(lambda a,b: b)

def get_images(file_path):
    img = tf.io.read_file(data_dir + file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (64, 64))
    return img

images_ds = fnames_ds.map(get_images)

labels_ds = labels_ds.map(lambda x: tf.one_hot(x, depth=10))

data_ds = tf.data.Dataset.zip((images_ds, labels_ds))

for item in data_ds:
    print(item)    


data_ds = data_ds.shuffle(buffer_size=20)
data_ds = data_ds.batch(batch_size=5)

model = Sequential([
    Input(shape=(64,64,3)),
    Conv2D(64, (5,5), activation="relu"),
    Flatten(),
    Dense(10, activation="relu")])

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

history = model.fit(data_ds, epochs=10)


#%% KERAS DATA GENERATORS
img_gen = ImageDataGenerator()

labels_df = pd.read_csv(csv_dir, header=0)

gen_iter = img_gen.flow_from_dataframe(
    dataframe=labels_df,
    directory=data_dir,
    x_col="file",
    y_col="label",
    class_mode="raw",
    batch_size=5,
    target_size=(64,64)
    )

for item in gen_iter:
    print(item)
    break

