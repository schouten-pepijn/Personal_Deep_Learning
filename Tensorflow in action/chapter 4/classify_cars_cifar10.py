import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import os


#%% DATA IMPORT
def unpickle(folder_path, file_names):
    data_list, label_list = [], []
    for file_name in file_names:
        file = os.path.join(folder_path, file_name)
        
        with open(file, 'rb') as fo:
            pfile = pickle.load(fo, encoding='bytes')
    
        data_list.append(pfile[b'data'])
        label_list.append(pfile[b'labels'])
    
    data, labels = np.array(data_list), np.array(label_list)
    data = data.reshape(-1, data.shape[-1])
    labels = labels.flatten()
    
    print(data.shape, labels.shape)
    input()
    
    return data

folder_path = "/Users/pepijnschouten/Desktop/Tensorflow in action/chapter 4/cifar-10-batches-py/"
file_names = ["data_batch_1", "data_batch_2",
              "data_batch_3", "data_batch_4",
              "data_batch_5"]

batch_1 = unpickle(folder_path, file_names)

#%%
test = df_train["img"][0]


#%% DATA FORMATTING
def format_data(x, depth):
    return (tf.dtypes.cast(x["img"]["bytes"], 'float32'),
            tf.one_hot(x["label"], depth=depth))

train_data = format_data(df_train, depth=10)
# test_data = df
