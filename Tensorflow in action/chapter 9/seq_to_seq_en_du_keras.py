import os 
os.chdir("/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow"
         " in action/chapter 9")
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from scripts.bleu import compute_bleu

from tqdm import tqdm
import json


#%% GLOBAL VARIABLES
RANDOM_SEED = 87
N_SAMPLES = 79368
START_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"
TEST_SIZE = 0.1
VALID_SIZE = 0.1
N_FREQ = 10
EN_SEQ_LEN = 19
TARGET_SEQ_LEN = 21

EPOCHS = 5
BATCH_SIZE = 128


#%% DATA PROCESSING
dutch_path = os.path.join("data", "ENGLISH_TO_DUTCH_TXT", "nld.txt")
german_path = os.path.join("data", "ENGLISH_TO_GERMAN_TXT", "deu.txt") 

df = pd.read_csv(dutch_path, delimiter='\t', header=None)
df.columns = ["English", "Target", "Attribution"]
df = df.drop(columns="Attribution")

print(f"df.shape = {df.shape}")

# clean problematic characters
clean_inds = [i for i in range(len(df))
              if b"\xc2" not in df.iloc[i]["Target"].encode("utf-8")]
df = df.iloc[clean_inds]

print(df.head())
print(df.tail())

# sample from whole data set
df = df.sample(n=N_SAMPLES, random_state=RANDOM_SEED)

# add start and end tokens
df["Target"] = START_TOKEN + ' ' + df["Target"] + ' ' + END_TOKEN

# split in train, valid and test
def train_valid_test_split(df, valid_size, test_size):
    not_train_size = valid_size + test_size
    valid_test_size = test_size / not_train_size
    train_df, not_train_df = train_test_split(df, test_size=not_train_size,
                                         shuffle=True)
    valid_df, test_df = train_test_split(not_train_df, test_size=valid_test_size,
                                         shuffle=True)
    
    return train_df, valid_df, test_df


train_df, valid_df, test_df = train_valid_test_split(df, VALID_SIZE, TEST_SIZE)

print(train_df.info())

#%% ANALYZE THE DATA
# vocab analysis
def get_vocab_size_greater_than(df, key, verbose=True):
    words = df[key].str.split().sum()
    cnt = Counter(words)
    
    freq_df = pd.Series(
        list(cnt.values()),
        index=list(cnt.keys())).sort_values(ascending=False)
    
    if verbose:
        print("\n", freq_df.head(n=10))
    
    n_vocab = (freq_df>=N_FREQ).sum()
    
    if verbose:
        print(f"\nVocab size: (>={N_FREQ} frequent): {n_vocab}")
        
    return n_vocab

print("\nEnglish corpus")
print("="*50)
en_vocab = get_vocab_size_greater_than(train_df, "English", verbose=True)

print("\nTarget vocab")
print('='*50)
target_vocab = get_vocab_size_greater_than(train_df, "Target", verbose=True)

# sequence length analysis
def print_seq_length(col_df):
    seq_length_ser = col_df.str.split(' ').str.len()
    
    print("\nSummary statistics")
    print(f"Median length: {seq_length_ser.median()}\n")
    print(seq_length_ser.describe())
    
    print("\nComputing the stats between the 1% and 99% quantiles (prevent outliers)")
    
    p_01 = seq_length_ser.quantile(0.01)
    p_99 = seq_length_ser.quantile(0.99)
    
    print(seq_length_ser[(seq_length_ser >= p_01) & (seq_length_ser < p_99)].describe())


print("\nEnglish corpus")
print("="*50)
print_seq_length(train_df["English"])

print("\nTarget corpus")
print("="*50)
print_seq_length(train_df["Target"])


#%% DEFINING SEQ TO SEQ MODEL (TEACHER FORCED)
# english goes in encoder, context vector as initiol state of decoder, 
# target language goes in decoder, decoder predicts next word via probabilities
def get_vectorizer(corpus, n_vocab, max_length=None,
                   return_vocab=True, name=None):
    
    inp = keras.Input(shape=(1,), dtype=tf.string, name='encoder_input')
    
    vectorize_layer = TextVectorization(
        max_tokens=n_vocab + 2, # due to padding and unknown token
        output_mode='int',
        output_sequence_length=max_length)
    
    vectorize_layer.adapt(corpus)
    vectorized_out = vectorize_layer(inp)
    
    if not return_vocab:
        return models.Model(
            inputs=inp, outputs=vectorized_out, name=name)
    else:
        return models.Model(
            inputs=inp, outputs=vectorized_out, name=name), vectorize_layer.get_vocabulary()


en_vectorizer, en_vocabulary = get_vectorizer(
    corpus=np.array(train_df["English"].tolist()),
    n_vocab=en_vocab,
    max_length=EN_SEQ_LEN,
    name='en_vectorizer')

target_vectorizer, target_vocabulary = get_vectorizer(
        corpus=np.array(train_df["Target"].tolist()),
        n_vocab=target_vocab,
        max_length=TARGET_SEQ_LEN-1,
        name='target_vectorizer')
    

def get_encoder(n_vocab, vectorizer):
    inp = keras.Input(shape=(1,), dtype=tf.string, name='e_input')
    vectorized_out = vectorizer(inp)
    
    emb_layer = layers.Embedding(
        n_vocab + 2, 128, mask_zero=True, name='e_embedding')
    emb_out = emb_layer(vectorized_out)
    
    gru_layer = layers.Bidirectional(
        layers.GRU(128, name='e_gru'), name='e_bidirectional_gru')
    gru_out = gru_layer(emb_out)
    
    encoder = models.Model(
        inputs=inp, outputs=gru_out, name='encoder')
    
    return encoder


encoder_exmpl = get_encoder(en_vocab, en_vectorizer)

print(encoder_exmpl.summary())

def get_final_seq2seq_model(n_vocab, encoder, vectorizer):
    e_inp = keras.Input(shape=(1,), dtype=tf.string, name="e_input_final")
    
    d_init_state = encoder(e_inp) # context vector
    
    d_inp = keras.Input(shape=(1,), dtype=tf.string, name="d_input")
    d_vectorized_out = vectorizer(d_inp)
    
    d_emb_layer = layers.Embedding(
        n_vocab + 2, 128, mask_zero=True, name='d_embedding')
    d_emb_out = d_emb_layer(d_vectorized_out)
    
    d_gru_layer = layers.GRU(
        256, return_sequences=True, name='d_gru')
    d_gru_out = d_gru_layer(d_emb_out, initial_state=d_init_state)
    
    d_dense_layer_1 = layers.Dense(
        512, activation='relu', name='d_dense_1')
    d_dense1_out = d_dense_layer_1(d_gru_out)
    
    d_dense_layer_final = layers.Dense(
        n_vocab+2, activation='softmax', name='d_dense_final')
    d_final_out = d_dense_layer_final(d_dense1_out)
    
    seq2seq_model = models.Model(
        inputs=[e_inp, d_inp], outputs=d_final_out,
        name='final_seq2seq')
    
    return seq2seq_model
    
seq2seq_exmpl = get_final_seq2seq_model(n_vocab=target_vocab,
                                       encoder=encoder_exmpl,
                                       vectorizer=target_vectorizer)
print(seq2seq_exmpl.summary())

    
#%% DEFINING FINAL PARAMETERS
en_vectorizer, en_vocabulary = get_vectorizer(
    corpus=np.array(train_df['English'].tolist()),
    n_vocab=en_vocab, max_length=EN_SEQ_LEN,
    name='e_vectorizer')

target_vectorizer, target_vocabulary = get_vectorizer(
    corpus=np.array(train_df['Target'].tolist()),
    n_vocab=target_vocab, max_length=TARGET_SEQ_LEN,
    name='d_vectorizer')

encoder = get_encoder(en_vocab, en_vectorizer) 

final_seq2seq_model = get_final_seq2seq_model(target_vocab,
                                              encoder,
                                              target_vectorizer)


#%% COMPILING THE MODEL
optimizer = Adam()
criterion = SparseCategoricalCrossentropy()
metrics = SparseCategoricalAccuracy()
final_seq2seq_model.compile(
    loss=criterion, optimizer=optimizer, metrics=[metrics])


#%% TRAINING AND EVALUATION HELPER FUNCTIONS
def prepare_data(train_df, valid_df, test_df):
    data_dict = dict()
    for label, df in zip(
            ['train', 'valid', 'test'], [train_df, valid_df, test_df]):
        en_inputs = np.array(df["English"].tolist())
        target_inputs = np.array(df['Target'].tolist())
        
        target_labels = np.array(
            df['Target'].str.split(n=1, expand=True).iloc[:, 1].tolist())
        
        data_dict[label] = {
            'encoder_inputs': en_inputs,
            'decoder_inputs': target_inputs,
            'decoder_labels': target_labels}
    
    return data_dict


def shuffle_data(en_inputs, target_inputs, target_labels, shuffle_indices=None):
    if shuffle_indices is None:
        shuffle_indices = np.random.permutation(np.arange(en_inputs.shape[0]))
    else:
        shuffle_indices = np.random.permutation(shuffle_indices)
    
    shuffled_data = (
        en_inputs[shuffle_indices],
        target_inputs[shuffle_indices],
        target_labels[shuffle_indices])

    return shuffled_data, shuffle_indices    


# define the BLEU metric
class BLEUMetric(object):
    def __init__(self, vocabulary, name='perplexity', **kwargs):
        super().__init__()
        self.vocab = vocabulary[2:]
        self.id_to_token_layer = layers.StringLookup(
            vocabulary=self.vocab, invert=True, num_oov_indices=0)
        
    def calculate_blue_from_preds(self, real, pred):
        pred_argmax = tf.argmax(pred, axis=1)
        pred_tokens = self.id_to_token_layer(pred_argmax)
        real_tokens = self.id_to_token_layer(real)
        
        def clean_text(tokens):
            t = tf.strings.strip(
                tf.strings.regex_replace(
                    tf.strings.join(
                        tf.transpose(tokens), separator=' '),
                    "eos.*", ''))
            
            t = np.char.decode(t.numpy().astype(np.bytes_), encoding='utf-8')
            t = [doc if len(doc) > 0 else '[UNK]' for doc in t]
            t = np.char.split(t).tolist()
            
            return t
        
        pred_tokens = clean_text(pred_tokens)
        real_tokens = [[r] for r in clean_text(real_tokens)]
        
        bleu, precisions, bp, ratio, translation_len, reference_len = compute_bleu(
            real_tokens, pred_tokens, smooth=False)
        
        return bleu
        
        
def eval_model(
        model, vectorizer, en_inputs_raw, target_inputs_raw, target_labels_raw,
        batch_size):
    bleu_metric = BLEUMetric(target_vocabulary)
    
    en_inputs_raw = tf.constant(en_inputs_raw)
    target_inputs_raw = tf.constant(target_inputs_raw)
    
    loss_log, accuracy_log, bleu_log = [], [], []
    n_batches = en_inputs_raw.shape[0] // batch_size
    print(" ", end='\r')
    
    for i in tqdm(range(n_batches)):
        
        x = [en_inputs_raw[i * batch_size:(i+1)*batch_size],
             target_inputs_raw[i * batch_size:(i+1)*batch_size]]
        
        y = vectorizer(target_labels_raw[i * batch_size:(i+1)*batch_size])
        
        loss, acc = model.evaluate(x, y, verbose=0)
        pred_y = model.predict(x)
        bleu = bleu_metric.calculate_blue_from_preds(y, pred_y)
        
        loss_log.append(loss)
        accuracy_log.append(acc)
        bleu_log.append(bleu)
        
    return np.mean(loss_log), np.mean(accuracy_log), np.mean(bleu_log)


def train_model(model, vectorizer, train_df, valid_df, test_df,
                epochs, batch_size):
    bleu_metric = BLEUMetric(target_vocabulary)
    
    data_dict = prepare_data(train_df, valid_df, test_df)
    
    shuffle_idx = None
    
    for epoch in range(epochs):
        bleu_log, accuracy_log, loss_log = [], [], []
        
        (en_inputs_raw, target_inputs_raw, target_labels_raw), shuffle_idx = shuffle_data(
            data_dict['train']['encoder_inputs'],
            data_dict['train']['decoder_inputs'],
            data_dict['train']['decoder_labels'],
            shuffle_idx)
        
        n_train_batches = en_inputs_raw.shape[0] // batch_size
        
        en_inputs_raw = tf.constant(en_inputs_raw)
        target_inputs_raw = tf.constant(target_inputs_raw)
        
        for i in tqdm(range(n_train_batches)):
            
            x = [en_inputs_raw[i * batch_size:(i+1)*batch_size],
                 target_inputs_raw[i * batch_size:(i+1)*batch_size]]
            y = vectorizer(target_labels_raw[i * batch_size:(i+1)*batch_size])
            
            model.train_on_batch(x, y)
            loss, acc = model.evaluate(x, y, verbose=0)
            pred_y = model.predict(x)
            bleu = bleu_metric.calculate_blue_from_preds(y, pred_y)
            
            loss_log.append(loss)
            accuracy_log.append(acc)
            bleu_log.append(bleu)
        
        val_en_inputs = data_dict['valid']['encoder_inputs']
        val_target_inputs = data_dict['valid']['decoder_inputs']
        val_target_labels = data_dict['valid']['decoder_labels']
        
        val_loss, val_acc, val_bleu = eval_model(model,
                                                 vectorizer,
                                                 val_en_inputs,
                                                 val_target_inputs,
                                                 val_target_labels,
                                                 batch_size)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(
            f"\ttrain loss: {np.mean(loss_log)} - accuracy: {np.mean(accuracy_log)}"
            f" - bleu: {np.mean(bleu_log)}")
        print(
            f"\tvalid loss: {val_loss} - accuracy: {val_acc}"
            f" - bleu: {val_bleu}")

        
        test_en_inputs = data_dict['test']['encoder_inputs']
        test_target_inputs = data_dict['test']['decoder_inputs']
        test_target_labels = data_dict['test']['decoder_labels']
        
        test_loss, test_acc, test_bleu = eval_model(model,
                                                 vectorizer,
                                                 test_en_inputs,
                                                 test_target_inputs,
                                                 test_target_labels,
                                                 batch_size)
        
        print(
            f"\ttest loss: {test_loss} - accuracy: {test_acc}"
            f" - bleu: {test_bleu}")
        

#%% TRAIN THE MODEL
train_model(final_seq2seq_model,
            target_vectorizer, 
            train_df,
            valid_df,
            test_df,
            EPOCHS,
            BATCH_SIZE)

os.makedirs('models', exist_ok=True)
models.save_model(final_seq2seq_model, os.path.join('models', 'seq2seq'))
with open(os.path.join('models', 'seq2seq_vocab', 'en_vocab.json')) as f:
    json.dump(en_vocabulary, f)
with open(os.path.join('models', 'seq2seq_vocab', 'target_vocab.json')) as f:
    json.dump(target_vocabulary, f)
    
#  p424