# tensorboard --logdir dir 
import os
os.chdir("/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow in action/chapter 12")
import tensorflow_datasets as tfds
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from collections import Counter
from tensorboard.plugins import projector


#%% DATA
fashion_ds = tfds.load('fashion_mnist')
print(fashion_ds)

def get_train_valid_test_ds(fashion_ds, batch_size, flatten_images=False):
    train_ds = fashion_ds['train'].shuffle(batch_size*20).map(
        lambda xy: (xy['image'], tf.reshape(xy['label'], [-1])))
    
    test_ds = fashion_ds['test'].shuffle(batch_size*20).map(
        lambda xy: (xy['image'], tf.reshape(xy['label'], [-1])))
    
    if flatten_images:
        train_ds = train_ds.map(lambda x, y: (tf.reshape(x, [-1]), y))
        test_ds = test_ds.map(lambda x, y: (tf.reshape(x, [-1]), y))
        
    valid_ds = train_ds.take(10000).batch(batch_size)
    train_ds = train_ds.skip(10000).batch(batch_size)
    
    return train_ds, valid_ds, test_ds

train_ds, valid_ds, test_ds = get_train_valid_test_ds(fashion_ds, batch_size=32)

id2label_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2:"Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"}


#%% DENSE MODEL
# logging dir
log_datetimestamp_format = "%Y%m%d%H%M%S"
log_datetimestamp = datetime.strftime(datetime.now(), log_datetimestamp_format)
image_logdir = "./logs/data_{}/train".format(log_datetimestamp)

# create image writer
image_writer = tf.summary.create_file_writer(image_logdir)

# write images to tensorboard
with image_writer.as_default():
    for data in fashion_ds["train"].batch(1).take(10):
        tf.summary.image(
            id2label_map[int(data["label"].numpy()[0])],
            data["image"],
            max_outputs=10,
            step=0)
        
# Write a batch of 20 images at once
with image_writer.as_default():
    for data in fashion_ds["train"].batch(20).take(1):
        pass
    tf.summary.image("A training data batch", data["image"],
                     max_outputs=20, step=0)
    
# model tracking
dense_model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')])

dense_model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',
                    metrics=['accuracy'])

# logging dir
log_datetimestamp_format = "%Y%m%d%H%M%S"
log_datetimestamp = datetime.strftime(
    datetime.now(), log_datetimestamp_format)
dense_log_dir = os.path.join("logs","dense_{}".format(log_datetimestamp))

# train parameters
batch_size = 64
tr_ds, v_ds, ts_ds = get_train_valid_test_ds(
    fashion_ds, batch_size=batch_size, flatten_images=True)

# tensorboard callback
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True,
    write_images=False, write_steps_per_second=False, update_freq='epoch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None,)

# train the model
dense_model.fit(tr_ds, validation_data=v_ds, epochs=10, callbacks=[tb_callback])


#%% CONV MODEL
conv_model = models.Sequential([
    layers.Conv2D(
            filters=32,
            kernel_size=(5,5),
            strides=(2,2),
            padding='same',
            activation='relu',
            input_shape=(28,28,1)),
    layers.Conv2D(
            filters=16,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation='relu'),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')])

conv_model.compile(
    loss="sparse_categorical_crossentropy", optimizer='adam',
    metrics=['accuracy'])

print(conv_model.summary())

# logging dir
log_datetimestamp_format = "%Y%m%d%H%M%S"
log_datetimestamp = datetime.strftime(
    datetime.now(), log_datetimestamp_format)
conv_log_dir = os.path.join("logs","conv_{}".format(log_datetimestamp))

# train parameters
batch_size = 64
tr_ds, v_ds, ts_ds = get_train_valid_test_ds(
    fashion_ds, batch_size=batch_size, flatten_images=False)

# tensorboard callback
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=conv_log_dir, histogram_freq=2, profile_batch=0)

# train the model
conv_model.fit(
    tr_ds, validation_data=v_ds, epochs=10, callbacks=[tb_callback])


#%% COMPARE EFFECTS OF BATCH NORM - CUSTOM METRICS
K.clear_session()

# without BN
dense_model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu', name='log_layer'),
    layers.Dense(10, activation='softmax')
    ]
)

dense_model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',
                    metrics=['accuracy']
)

print(dense_model.summary())

# with BN
dense_model_bn = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu', name='log_layer_bn'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
    ]
)

dense_model_bn.compile(
    loss="sparse_categorical_crossentropy", optimizer='adam',
    metrics=['accuracy']
)

print(dense_model_bn.summary())

# custom train loop
def train_model(model, dataset, log_dir, log_layer_name, epochs):
    # define the writer
    writer = tf.summary.create_file_writer(log_dir)
    
    step = 0
    
    # open the writer
    with writer.as_default():
        for e in range(epochs):
            print(f"Training epoch: {e+1}")
            
            for b_num, batch in enumerate(dataset):
                
                # train one batch
                model.train_on_batch(*batch)
                
                # extract the weights
                weights = model.get_layer(log_layer_name).get_weights()[0]
                
                # log mean
                layer_mean = np.mean(np.abs(weights))
                tf.summary.scalar("mean_weights", layer_mean, step=step)
                # log std
                layer_std = np.std(np.abs(weights))
                tf.summary.scalar("std_weights", layer_std,
                                  step=step)
                
                # flush to the disk from the buffer
                writer.flush()
                
                # print statements
                if b_num % 20 == 0:
                    print(f"\tBatch: {b_num:3} | " 
                          f" Layer mean: {layer_mean:.5f} | "
                          f" Layer std: {layer_std:.5f}")
                
                step += 1
                
            print("\tDone")
            
    print("Training completed\n")
    
    
batch_size = 64
exp_log_dir = 'bn_exp_logs'

# train dense model
tr_ds, _, _ = get_train_valid_test_ds(
    fashion_ds, batch_size, flatten_images=True)

train_model(dense_model, tr_ds,
            log_dir=os.path.join(exp_log_dir, 'standard_dense'),
            log_layer_name='log_layer', epochs=5)

# train dense model with batch norm
tr_ds, _, _ = get_train_valid_test_ds(
    fashion_ds, batch_size, flatten_images=True)

train_model(dense_model, tr_ds,
            log_dir=os.path.join(exp_log_dir, 'batchnorm_dense'),
            log_layer_name='log_layer_bn', epochs=5)


#%% SHOWING WORD VECTORS IN TENSORBOARD
# by using GloVe vectors
glove_path = os.path.join('GloVe', 'glove.50d.txt')
df = pd.read_csv(
        glove_path,
        header=None, index_col=0, sep=None,
        on_bad_lines='skip',
        encoding='utf-8',
        engine='python')

print(df.head())

# load imdb review data
review_ds = tfds.load('imdb_reviews')
train_review_ds = review_ds['train']

# create corpus of review text
corpus = []
for data in train_review_ds:
    txt = str(np.char.decode(data['text'].numpy(),
                             encoding='utf-8')).lower()
    corpus.append(str(txt))

# select 5000 most common words in review data
corpus = ' '.join(corpus)
cnt = Counter(corpus.split())
most_common_words = [w for w, _ in cnt.most_common(5000)]
print(cnt.most_common(100))

# find common tokens
df_common = df.loc[df.index.isin(most_common_words)]

# show word vectors on tensorboard
log_dir = os.path.join('word_vector_logs', 'embeddings')
weights = tf.Variable(df_common.values)

#save embeddings as TF checkpoint
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join('word_vector_logs', 'embedding.ckpt'))

# save the meta data
with open(os.path.join('word_vector_logs', 'metadata.tsv'), 'w') as f:
    for w in df_common.index:
        f.write(w + '\n')
    
# create projector specific configuration
config = projector.ProjectorConfig()
embedding = config.embeddings.add()

# add meta data to the projection
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config) 

"""" open tensorflow at local host """
