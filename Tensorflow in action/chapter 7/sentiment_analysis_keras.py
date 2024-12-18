import os
os.chdir("/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow in action/chapter 7")
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import re
from tqdm import tqdm
from collections import Counter
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer

import nltk
# text cleaning libs
nltk.download("averaged_perceptron_tagger_eng", download_dir="nltk")
nltk.download("wordnet", download_dir="nltk")
nltk.download("omw-1.4", download_dir="nltk")
nltk.download("stopwords", download_dir="nltk")
nltk.download("punkt_tab", download_dir="nltk")
nltk.download("averaged_perceptron_tagger_eng", download_dir="nltk")
nltk.data.path.append(os.path.abspath("nltk"))

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.tag import pos_tag
import string

os.environ["KERAS_BACKEND"] = """'jax'""" 'tensorflow'
#%% VARIABLES
RANDOM_STATE = 87
BATCH_SIZE = 64
EPOCHS = 1

# remove stopwords and punctuation
EN_STOPWORDS = set(stopwords.words("english"))
EN_STOPWORDS.remove("not")
EN_STOPWORDS.remove("no")
EN_STOPWORDS.add("s")
EN_STOPWORDS.add("...")


#%% DATA IMPORT
path = os.path.join("data", "Video_Games_5.json")
review_df = pd.read_json(path, lines=True, orient="records")
review_df = review_df[["overall", "verified", "reviewTime", "reviewText"]]

print(review_df.info())
print(review_df.head())


#%% DATA PROCESSING
# remove missing review text
review_df = review_df[~review_df["reviewText"].isna()]
review_df = review_df[review_df["reviewText"].str.strip().str.len()>0]

# only consider verified buyers
print(review_df["verified"].value_counts())
verified_df = review_df[review_df["verified"]].copy()

# check class imbalance
print(verified_df["overall"].value_counts())

plt.figure(dpi=100)
sns.histplot(data=verified_df, x="overall",
             bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.show()

# group reviews in positive or negative
verified_df["label"] = verified_df["overall"].map(
    {5: 1, 4: 1, 3: 0, 2: 0, 1: 0})
print(verified_df["label"].value_counts())

# shuffle the data
verified_df = verified_df.sample(frac=1.0, random_state=RANDOM_STATE)

# split features and labels
features, labels = verified_df["reviewText"], verified_df["label"]


#%% TEXT CLEANING
def clean_text(doc):
    # make lower case
    doc = doc.lower()
    
    # expand n't to not and replace shortened forms
    doc = re.sub(pattern=r"\w+n\'t ", repl="not ", string=doc)
    doc = re.sub(pattern=r"(?:\'ll |\'re |\'d |\'ve )", repl=" ", string=doc)
    doc = re.sub(pattern=r"/d+", repl="", string=doc)
    
    tokens = [w for w in word_tokenize(doc) if w not in EN_STOPWORDS
              and w not in string.punctuation]
    
    # lemmantize the text
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    clean_text = [lemmatizer.lemmatize(w, pos=p[0].lower()) \
                  if p[0] == "N" or p[0] == "V" else w \
                      for (w, p) in pos_tags]
    return clean_text

# Sample features
sample_doc = "She sells seashells by the seashore."
print(f"Before clean: {sample_doc}")
print(f"After clean: {clean_text(sample_doc)}")

# apply data cleaning to all features
tqdm.pandas()
print("Cleaning text")
features = features.progress_apply(lambda x: clean_text(x))


#%% DATA PREPARATION
def train_valid_test_split(features, labels, train_frac=0.8):
    neg_indices = pd.Series(labels.loc[(labels==0)].index)
    pos_indices = pd.Series(labels.loc[(labels==1)].index)
    
    n_valid = int(min(
        [len(neg_indices), len(pos_indices)]) * ((1-train_frac) / 2.0))
    n_test = n_valid
    
    neg_test_idx = neg_indices.sample(n=n_test, random_state=RANDOM_STATE)
    neg_valid_idx = neg_indices.loc[-neg_indices.isin(neg_test_idx)].sample(n=n_test,
                                                                            random_state=RANDOM_STATE)
    neg_train_idx = neg_indices.loc[-neg_indices.isin(
        neg_test_idx.tolist() + neg_valid_idx.tolist())]
    
    pos_test_idx = pos_indices.sample(n=n_test, random_state=RANDOM_STATE)
    pos_valid_idx = pos_indices.loc[-pos_indices.isin(pos_test_idx)].sample(n=n_test,
                                                                            random_state=RANDOM_STATE)
    pos_train_idx = pos_indices.loc[-pos_indices.isin(
        pos_test_idx.tolist() + pos_valid_idx.tolist())]
    
    tr_x = features.loc[neg_train_idx.tolist() + \
                        pos_train_idx.tolist()].sample(frac=1.0, random_state=RANDOM_STATE)
    tr_y = labels.loc[neg_train_idx.tolist() + \
                      pos_train_idx.tolist()].sample(frac=1.0, random_state=RANDOM_STATE)
    v_x = features.loc[neg_valid_idx.tolist() + \
                       pos_valid_idx.tolist()].sample(frac=1.0, random_state=RANDOM_STATE)
    v_y = labels.loc[neg_valid_idx.tolist() + \
                     pos_valid_idx.tolist()].sample(frac=1.0, random_state=RANDOM_STATE)
    ts_x = features.loc[neg_test_idx.tolist() + \
                        pos_test_idx.tolist()].sample(frac=1.0, random_state=RANDOM_STATE)
    ts_y = labels.loc[neg_test_idx.tolist() + \
                      pos_test_idx.tolist()].sample(frac=1.0, random_state=RANDOM_STATE)
        
    print(f"Training data: {len(tr_x)}")
    print(f"Validation data: {len(v_x)}")
    print(f"Test data: {len(ts_x)}")
    
    return (tr_x, tr_y), (v_x, v_y), (ts_x, ts_y)

(tr_x, tr_y), (v_x, v_y), (ts_x, ts_y) = train_valid_test_split(features, labels)


#%% ANALYZE THE VOCAB
# word frequency dictionary
data_list = [w for doc in tr_x for w in doc]
cnt = Counter(data_list)
freq_df = pd.Series(
    list(cnt.values()),
    index=list(cnt.keys())).sort_values(ascending=False)

print(freq_df.head(n=10))
print(freq_df.describe())

# vocab count of words with n>=25
n_vocab = (freq_df >= 25).sum()
print(n_vocab)


#%% ANALYZE THE SEQUENCE LENGTH
seq_len_ser = tr_x.str.len()

# filter outlier words based on quantiles
p_10 = seq_len_ser.quantile(0.1)
p_90 = seq_len_ser.quantile(0.9)

# bin review to short, medium and long
print(seq_len_ser[(seq_len_ser >= p_10) & (seq_len_ser <= p_90)].describe(
        percentiles=[0.33, 0.66])) # short < [0,6), medium [5,16) long [16,inf)


#%% TEXT TO WORDS AND WORDS TO NUMBERS
tokenizer = Tokenizer(
    num_words=n_vocab,
    oov_token="unk",
    lower=True,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    split=" ",
    char_level=False)

# learn word to number mapping
tokenizer.fit_on_texts(tr_x.tolist())

# convert text corpus to seq of indices
tr_x = tokenizer.texts_to_sequences(tr_x.tolist())
v_x = tokenizer.texts_to_sequences(v_x.tolist())
ts_x = tokenizer.texts_to_sequences(ts_x.tolist())


#%% INVESTIGATE LOOK-UP TABLE
index_word = {v: k for k, v in tokenizer.word_index.items()}
def sequences_to_texts(list_of_idx):
    words = [index_word.get(word) for word in list_of_idx]
    return words

text_example = list(map(sequences_to_texts, tr_x))

for i in range(3):
    print(text_example[i], tr_x[i], "", sep="\n")


#%% CREATE DATA PIPELINE
def get_tf_pipeline(
        text_seq, labels, batch_size=64, bucket_bounds=[5, 16],
        max_length=50, shuffle=False):
    data_seq = [[b] + a for a, b in zip(text_seq, labels)]
    
    tf_data = tf.ragged.constant(data_seq)[:, :max_length]
    
    text_ds = tf.data.Dataset.from_tensor_slices(tf_data)
    text_ds = text_ds.filter(lambda x: tf.size(x) > 1)
    
    bucket_fn = tf.data.experimental.bucket_by_sequence_length(
        element_length_func=lambda x: tf.cast(tf.shape(x)[0], "int32"),
        bucket_boundaries=bucket_bounds,
        bucket_batch_sizes=[batch_size, batch_size, batch_size],
        padded_shapes=None,
        padding_values=0,
        pad_to_bucket_boundary=False)
    
    text_ds = text_ds.map(lambda x: x).apply(bucket_fn)
    
    if shuffle:
        text_ds = text_ds.shuffle(buffer_size=10*batch_size)
        
    text_ds = text_ds.map(lambda x: (x[:, 1:], x[:, 0]))
    
    return text_ds
    
text_ds = get_tf_pipeline(tr_x, tr_y)

print(next(iter(text_ds)))

#%% ONE HOT MODEL FOR SENTIMENT ANALYSIS
class OneHotEncoderLayer(keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHotEncoderLayer, self).__init__(**kwargs)
        self.depth = depth
        
    def build(self, input_shape):
        pass
    
    def call(self, inputs):
        inputs = keras.ops.cast(inputs, "int32")
        
        if len(inputs.shape) == 3:
            inputs = inputs[..., 0]
        # one hot encoding as a simple sub for word embeddings
        return keras.ops.one_hot(inputs, num_classes=self.depth)
    
    def compute_mask(self, inputs, mask=None):
        return mask
    
    def get_config(self):
        base_config = super().get_config().copy()
        config ={"depth": self.depth}
        return {**base_config, **config}
            

keras.backend.clear_session()

model_oh = keras.models.Sequential([
    keras.layers.Input(shape=(None, 1)),
    keras.layers.Masking(mask_value=0.0),
    OneHotEncoderLayer(depth=n_vocab),
    keras.layers.LSTM(128, return_state=False, return_sequences=False),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
    ])

criterion = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam()
metrics = keras.metrics.BinaryAccuracy(threshold=0.5)
model_oh.compile(loss=criterion,
                 optimizer=optimizer,
                 metrics=[metrics])

print(model_oh.summary())


#%% TRAINING THE ONE HOT MODEL
# create datasets
train_ds = get_tf_pipeline(tr_x, tr_y, batch_size=BATCH_SIZE, shuffle=True)
valid_ds = get_tf_pipeline(v_x, v_y, batch_size=BATCH_SIZE, shuffle=False)

# define weighting factor for class imbalance
neg_weight = (tr_y == 1).sum() / (tr_y == 0).sum()

# create callbacks and logging dir
os.makedirs("eval", exist_ok=True)
csv_logger = keras.callbacks.CSVLogger(
    os.path.join("eval", "model_oh_sentiment_analysis.log"))

monitor_metric = "val_loss"
mode = "min"

lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor=monitor_metric, factor=0.1, patience=3, mode=mode, min_lr=1e-8)

es_callback = keras.callbacks.EarlyStopping(
    monitor=monitor_metric, patience=6, mode=mode, restore_best_weights=False)

cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join("eval", "model_oh_cp.keras"), save_weights_only=False,
    verbose=1)

# fit the model
model_oh.fit(
    x=train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    # steps_per_epoch=100,
    class_weight={0: neg_weight, 1: 1.0},
    callbacks=[es_callback, lr_callback, cp_callback, csv_logger])

# save the model
model_oh.save(os.path.join("eval", "final_sentiment_model_oh.keras"))


#%% TESTING THE ONE HOT MODEL
test_ds = get_tf_pipeline(ts_x, ts_y, batch_size=BATCH_SIZE, shuffle=False)

model_oh.evaluate(test_ds)

""" IMPROVED MODEL"""
#%% THE EMBEDDING MODEL FOR SENTIMENT ANALYSIS
model_emb = keras.models.Sequential([
    keras.layers.Input(shape=(None,)),
    keras.layers.Embedding(input_dim=n_vocab+1, output_dim=128, mask_zero=True),
    keras.layers.LSTM(128, return_state=False, return_sequences=False),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
    ])

print(model_emb.summary())

model_emb.compile(loss=criterion,
                  optimizer=optimizer,
                  metrics=[metrics])


#%% TRAINING THE EMB MODEL
# create datasets
train_ds = get_tf_pipeline(tr_x, tr_y, batch_size=BATCH_SIZE, shuffle=True)
valid_ds = get_tf_pipeline(v_x, v_y, batch_size=BATCH_SIZE, shuffle=False)

# callbacks and logger dir
csv_logger = keras.callbacks.CSVLogger(
    os.path.join("eval", "model_emb_sentiment_analysis.log"))

monitor_metric = "val_loss"
mode = "min"

lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor=monitor_metric, factor=0.1, patience=3, mode=mode, min_lr=1e-8)

es_callback = keras.callbacks.EarlyStopping(
    monitor=monitor_metric, patience=6, mode=mode, restore_best_weights=False)

cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join("eval", "model_emb_cp.keras"), save_weights_only=False,
    verbose=1)

# fit the model
model_oh.fit(
    x=train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    # steps_per_epoch=100,
    class_weight={0: neg_weight, 1: 1.0},
    callbacks=[es_callback, lr_callback, cp_callback, csv_logger])

# save the model
model_oh.save(os.path.join("eval", "final_sentiment_model_emb.keras"))


#%% TESTING EMB MODEL
test_ds = get_tf_pipeline(ts_x, ts_y, batch_size=BATCH_SIZE, shuffle=False)

model_oh.evaluate(test_ds)


#%% TEST THE TOP k NEG AND POS REVIEWS
test_ds = get_tf_pipeline(ts_x, ts_y, batch_size=BATCH_SIZE, shuffle=False)

test_x, test_pred, test_y = [], [], []
for x, y in test_ds:
    test_x.append(x)
    test_pred.append(model_emb.predict(x))
    test_y.append(y)
    
test_x = [doc for t in test_x for doc in t.numpy().tolist()]
test_pred = tf.concat(test_pred, axis=0).numpy()
test_y = tf.concat(test_y, axis=0).numpy()

sorted_pred = np.argsort(test_pred.flatten())
min_pred, max_pred = sorted_pred[:5], sorted_pred[-5:]

print("\nMost negative reviews:\n")
print("="*50)
for i in min_pred:
    print(" ".join(tokenizer.sequences_to_texts([test_x[i]])), sep="\n")
    
print("\nMost positive reviews:\n")
print("="*50)
for i in max_pred:
    print(" ".join(tokenizer.sequences_to_texts([test_x[i]])), sep="\n")