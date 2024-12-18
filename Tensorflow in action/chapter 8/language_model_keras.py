import os
os.chdir("/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow in action/chapter 8")
import shutil
from collections import Counter
import pandas as pd
from itertools import chain
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import tensorflow.keras.ops as K
import numpy as np


#%% VARIABLE
PREVIOUS_MODEL = True
N_GRAMS = 2
N_SEQ = 100
N_FREQ = 10

BATCH_SIZE = 128
LR = 1e-8
EPOCHS = 50

TRAIN_MODEL = False
EVALUATE_MODEL = False

GENERATE_TEXT = True
N_GENERATED_GREEDY = 500
N_GENERATED_BEAM = 100

BEGINNING_TEXT = "CHAPTER I. Down the Rabbit-Hole Alice was beginning to get very tired of sitting by her sister on the bank ,"

#%% MOVE PREVIOUS TRAINING WEIGHTS AND FILES
if PREVIOUS_MODEL:
    # moving files
    source_folder_1 = 'models'
    source_folder_2 = 'eval'
    
    destination_folder_1 = os.path.join('archive', source_folder_1)
    destination_folder_2 = os.path.join('archive', source_folder_2)
    
    os.makedirs('archive', exist_ok=True)
    os.makedirs(destination_folder_1, exist_ok=True)
    os.makedirs(destination_folder_2, exist_ok=True)
    
    if os.path.exists(source_folder_1):
        for file_name in os.listdir(source_folder_1):
            source = os.path.join(source_folder_1, file_name)
            destination = os.path.join(destination_folder_1, file_name)
            shutil.move(source, destination)
            print(f"Moved: {file_name}")
        
    if os.path.exists(source_folder_2):
        for file_name in os.listdir(source_folder_2):
            source = os.path.join(source_folder_2, file_name)
            destination = os.path.join(destination_folder_2, file_name)
            shutil.move(source, destination)
            print(f"Moved: {file_name}")
        
    LOAD_PATH = os.path.join(destination_folder_1, f'{N_GRAMS}_gram_lm.h5')
    

#%% PATHS
FINAL_SAVE_PATH = os.path.join('models', f'{N_GRAMS}_gram_lm.h5')
TRAIN_PATH = "/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow in action/chapter 8/CHILD_BOOK_LANGUAGE_TXT/data/cbt_train.txt"
VAL_PATH = "/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow in action/chapter 8/CHILD_BOOK_LANGUAGE_TXT/data/cbt_valid.txt"
TEST_PATH = "/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow in action/chapter 8/CHILD_BOOK_LANGUAGE_TXT/data/cbt_test.txt"        


#%% DATA LOAD AND EXPLORATION
def read_data(path):
    stories = []
    with open(path, 'r') as f:
        s = []
        for row in f:
        
            if row.startswith("_BOOK_TITLE_"):
                if len(s) > 0:
                    stories.append(' '.join(s).lower())
                s = []
            else:
                s.append(row)
            
    if len(s) > 0:
        stories.append(' '.join(s).lower())
            
    return stories


train_stories = read_data(TRAIN_PATH)
val_stories = read_data(VAL_PATH)
test_stories = read_data(TEST_PATH)


data_list = [w for doc in train_stories for w in doc.split(' ')]
cnt = Counter(data_list)
freq_df = pd.Series(
    list(cnt.values()),
    index=list(cnt.keys())).sort_values(ascending=False)


print(freq_df.head())
print(f"Vocab size (>={N_FREQ} frequent): {(freq_df>=N_FREQ).sum()}")


#%% CREATE NGRAMS
def get_ngrams(text, n):
    return [text[i:i+n] for i in range(0, len(text), n)]

# testing ngrams
test_string = "I like chocolates"
print("Original: {}".format(test_string))
for i in list(range(3)):
    print(f"\t{i+1}-grams: {get_ngrams(test_string, i+1)}")
    
    
text = chain(*[get_ngrams(s, N_GRAMS) for s in train_stories])
cnt = Counter(text)

# Create a pandas series with the counter results
freq_df = pd.Series(
    list(cnt.values()),
    index=list(cnt.keys())).sort_values(ascending=False)


N_VOCAB = (freq_df>=N_FREQ).sum()
print(f"Size of vocabulary: {N_VOCAB}")


#%% TOKENIZE TEXT
tokenizer = Tokenizer(num_words=N_VOCAB, oov_token='unk', lower=True)
train_ngram_stories = [get_ngrams(s, N_GRAMS) for s in train_stories]
tokenizer.fit_on_texts(train_ngram_stories)
train_data_seq = tokenizer.texts_to_sequences(train_ngram_stories)

val_ngram_stories = [get_ngrams(s, N_GRAMS) for s in val_stories]
val_data_seq = tokenizer.texts_to_sequences(val_ngram_stories)

test_ngram_stories = [get_ngrams(s, N_GRAMS) for s in test_stories]
test_data_seq = tokenizer.texts_to_sequences(test_ngram_stories)


#%% TF.DATA PIPELINE
def get_tf_pipeline(data_seq, n_seq, batch_size=64, shift=1, shuffle=True):
    text_ds = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(data_seq))
   
    if shuffle:
        text_ds = text_ds.shuffle(buffer_size=len(data_seq)//2)
        
    text_ds = text_ds.flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(
            x
            ).window(
                n_seq+1, shift=shift
                ).flat_map(
                    lambda window: window.batch(n_seq+1, drop_remainder=True)))
                    
    if shuffle:
        text_ds = text_ds.shuffle(buffer_size=10*batch_size)
    
    text_ds = text_ds.batch(batch_size)
    
    text_ds = tf.data.Dataset.zip(
        text_ds.map(lambda x: (x[:,:-1], x[:, 1:]))).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
    
    return text_ds


ds = get_tf_pipeline(train_data_seq, 5, batch_size=6)

for a in ds.take(1):
    print(a)
    

#%% SAVE HYPERPARAMETERS
print(f"n_grams uses n = {N_GRAMS}")
print(f"Vocabulary size: {N_VOCAB}")
print(f"Sequence length for model: {N_SEQ}")

os.makedirs('models', exist_ok=True)
with open(os.path.join('models', 'text_hyperparams.pkl'), 'wb') as f:
    pickle.dump({'n_vocab': N_VOCAB, 'ngrams': N_GRAMS, 'n_seq': N_SEQ}, f)
    
    
#%% IMPREMENTING THE LANGUAGE MODEL
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(None,)),
    tf.keras.layers.Embedding(
        input_dim=N_VOCAB+1, output_dim=512, name='embedding'),
    tf.keras.layers.GRU(
        1024, return_state=False, return_sequences=True, name='gru'),
    tf.keras.layers.Dense(
        512, activation='relu', name='dense'),
    tf.keras.layers.Dense(N_VOCAB, name='final_out'),
    tf.keras.layers.Activation(activation='softmax')])

print(model.summary())


#%% EVALUATION METRICS
class PerplexityMetric(tf.keras.metrics.Mean):
    def __init__(self, name='perplexity', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')
        
    def _calculate_perplexity(self, real, pred):
        loss_ = self.cross_entropy(real, pred)
        mean_loss = K.mean(loss_, axis=-1)
        perplexity = K.exp(mean_loss)
        
        return perplexity

    def update_state(self, y_true, y_pred, sample_weight=None):
        perplexity = self._calculate_perplexity(y_true, y_pred)
        
        super().update_state(perplexity)
        
        
#%% COMPILE MODEL
if PREVIOUS_MODEL:
    model.load_weights(LOAD_PATH)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', PerplexityMetric()])


#%% TRAINING THE MODEL
if TRAIN_MODEL:
    train_ds = get_tf_pipeline(
        train_data_seq, N_SEQ, shift=25, batch_size=BATCH_SIZE)
    valid_ds = get_tf_pipeline(
        val_data_seq, N_SEQ, shift=N_SEQ, batch_size=BATCH_SIZE)

    # call backs
    os.makedirs('eval', exist_ok=True)
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join('eval','1_language_modelling.log'))
    
    monitor_metric = 'val_perplexity'
    mode = 'min'
    print(f"Using metric = {monitor_metric} and mode= {mode} for EarlyStopping")
    
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric, factor=0.1, patience=2, mode=mode, min_lr=LR)
    
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric, patience=5, mode=mode, restore_best_weights=False)
    
    
    ckp_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join('models', 'best_model_ckp.weights.h5'),
        save_weights_only=True,
        monitor=monitor_metric,
        mode=mode,
        save_best_only=True)

    # training
    model.fit(train_ds, epochs=EPOCHS, validation_data = valid_ds,
              callbacks=[es_callback, lr_callback,
                         csv_logger, ckp_callback])
    
    # saving
    os.makedirs('models', exist_ok=True)
    tf.keras.models.save_model(model, FINAL_SAVE_PATH)
 
    
#%% MODEL EVALUATION
if EVALUATE_MODEL:
    test_ds = get_tf_pipeline(
        test_data_seq, N_SEQ, shift=N_SEQ)
    eval_history = model.evaluate(test_ds)
    eval_history = {k: v for k, v in zip(
        ["Val loss", "Val accuracy", "Val perplexity"],
        eval_history)}
    print(eval_history)


#%% INFERENCE MODEL
def create_infer_model(train_model):
    # create evaluation model
    text_inp = tf.keras.layers.Input(shape=(None,)) 
    state_inp = tf.keras.layers.Input(shape=(1024,))
    
    emb_layer = tf.keras.layers.Embedding(
        input_dim=N_VOCAB+1, output_dim=512)
    emb_out = emb_layer(text_inp)
    
    gru_layer = tf.keras.layers.GRU(
        1024, return_state=True, return_sequences=True)
    gru_out, gru_state = gru_layer(emb_out,
                                   initial_state=state_inp)
    
    dense_layer = tf.keras.layers.Dense(512, activation='relu')
    dense_out = dense_layer(gru_out)
    
    final_dense_layer = tf.keras.layers.Dense(N_VOCAB, name='final_out')
    final_dense_out = final_dense_layer(dense_out)
    softmax_out = tf.keras.layers.Activation(
        activation='softmax')(final_dense_out)
    
    infer_model = tf.keras.models.Model(
        inputs=[text_inp,
                state_inp
                ]
        , outputs=[softmax_out, gru_state])
    
    # copy training weights
    emb_layer.set_weights(train_model.get_layer('embedding').get_weights())
    gru_layer.set_weights(train_model.get_layer('gru').get_weights())
    dense_layer.set_weights(train_model.get_layer('dense').get_weights())
    final_dense_layer.set_weights(
        train_model.get_layer('final_out').get_weights())
    
    return infer_model

if GENERATE_TEXT:
    infer_model = create_infer_model(model)
    print(infer_model.summary())

    
#%% TEXT PREDICTION - GREEDY DECODING
if GENERATE_TEXT:
    text = get_ngrams(BEGINNING_TEXT.lower(), N_GRAMS)
    
    seq = tokenizer.texts_to_sequences([text])
    state = np.zeros(shape=(1, 1024))
    
    print(tf.executing_eagerly())
    
    # process input text
    for c in seq[0]:
        print(c)
        out, state = infer_model.predict([np.array([[c]]), state])
        # print(out, state)
        # input()
        
        state = state[np.newaxis]
    
    # select last word for prediction
    wid = int(np.argmax(out[0], axis=-1).ravel()[0])
    word = tokenizer.index_word[wid]
    text.append(word)
    
    
    
    # predict N_GENERATED_NGRAMS new text
    x = np.array([[wid]])
    for _ in range(N_GENERATED_GREEDY):
        
        out, state = infer_model.predict([x, state])
        state = state[np.newaxis]
        
        out_argsort = np.argsort(out[0], axis=-1).ravel()
        wid = int(out_argsort[-1])
        word = tokenizer.index_word[wid]
        
        # break repeated text by selecting from top 3 likely outputs
        if word.endswith(' '):
            if np.random.normal() > 0.5:
                width = 3
                i = np.random.choice(
                    list(range(-width, 0)),
                    p=out_argsort[-width:] / out_argsort[-width:].sum())
                wid = int(out_argsort[i])
         
                word = tokenizer.index_word[wid]
                
        x = np.array([[wid]])
        text.append(word)
        
        if i % 20 == 0:
            print(f"\n-- Word iteration: {i} --\n")
    
    
    print('\n')
    print('='*60)
    print('Final text: ')
    print(''.join(text))


#%% TEXT PREDICTION - BEAM SEARCH
# use the joint probability of multiple predictions in the future at each time step
if GENERATE_TEXT:
    def beam_one_step(model, input_, state):
        output, new_state = model.predict([input_, state])
        new_state = new_state[np.newaxis]
        return output, new_state
    
    def beam_search(
            model, input_, state, beam_depth=5, beam_width=3, ignore_blank=True):
        
        results = []
        sequence = []
        log_prob = 0.0
        
        def recursive_fn(input_, state, sequence, log_prob, i):
            # termination of search
            if i == beam_depth:
                results.append((list(sequence), state, np.exp(log_prob)))
                return sequence, log_prob, state
            else:
                output, new_state = beam_one_step(model, input_, state)
                # top beam_width probabilities
                top_probs, top_ids = tf.nn.top_k(output, k=beam_width)
                top_probs, top_ids = top_probs.numpy().ravel(), top_ids.numpy().ravel()
                
                # for each candidate compute next pred
                for p, wid in zip(top_probs, top_ids):
                    new_log_prob = log_prob + np.log(p)
                    
                    # penalize repeated words
                    if len(sequence) > 0 and wid == sequence[-1]:
                        new_log_prob = new_log_prob + np.log(1e-1)
                    
                    sequence.append(wid)
                    # recursively call the function
                    _ = recursive_fn(
                        np.array([[wid]]), new_state, sequence, new_log_prob, i+1)
                    
                    sequence.pop()
                    
        recursive_fn(input_, state, sequence, log_prob, 0)
        results = sorted(results, key=lambda x: x[2], reverse=True)
        
        return results
    
    
    text = get_ngrams(BEGINNING_TEXT.lower(), N_GRAMS)
    
    seq = tokenizer.texts_to_sequences([text])
    state = np.zeros(shape=(1, 1024))
    
    # process input text
    for c in seq[0]:
        out, state = infer_model.predict([np.array([[c]]), state])
        state = state[np.newaxis]
    
    # select last word for prediction
    wid = int(np.argmax(out[0], axis=-1).ravel()[0])
    word = tokenizer.index_word[wid]
    text.append(word)
    
    # predict N_GENERATED_NGRAMS new text
    x = np.array([[wid]])
    for i in range(N_GENERATED_BEAM):
        result = beam_search(infer_model, x, state, 7, 2)
        
        n_probs = np.array([p for _, _, p in result[:10]])
        p_j = np.random.choice(list(range(n_probs.size)), p=n_probs/n_probs.sum())
        
        best_beam_ids, state, _ = result[p_j]
        x = np.array([[best_beam_ids[-1]]])
        
        text.extend([tokenizer.index_word[w] for w in best_beam_ids])
        
        if i % 20 == 0:
            print(f"\n-- Word iteration: {i} --\n")
        
    
    print('\n')
    print('='*60)
    print('Final text: ')
    print(''.join(text))