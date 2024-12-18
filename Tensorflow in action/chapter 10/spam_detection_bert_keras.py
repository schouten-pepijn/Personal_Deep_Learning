import os
os.chdir("/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow in action/chapter 10")
os.environ['TF_USE_LEGACY_KERAS']='1'
import numpy as np
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow_models as tfm
import tensorflow_hub as hub
import tensorflow as tf

#%% VARIABLES
n_class = 100
random_seed = 87

hub_bert_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
max_seq_length = 60

num_classes = 2

#%% DATA INPORT
txt_path = os.path.join("data", "SMS_SPAM_TXT", "SMSSpamCollection")

def load_data(txt_path):
    inputs, labels = [], []
    n_ham, n_spam = 0, 0
    
    with open(txt_path, 'r') as f:
        for r in f:
            # no-spam
            if r.startswith('ham'):
                label = 0
                txt = r[4:]
                n_ham += 1
            elif r.startswith('spam'):
                label = 1
                txt = r[5:]
                n_spam += 1
            
            inputs.append(txt)
            labels.append(label)
    
    inputs = np.array(inputs).reshape(-1, 1)
    labels = np.array(labels)
    
    return inputs, labels

inputs, labels = load_data(txt_path)


#%% TREATING CLASS IMBALACE BY UNDERSAMPLING MAJORITY CLASS
print("\nFull data set:")
print(f"\t% spam: {100 * labels.sum() / len(labels)}")
print(f"\t% ham: {100 * (len(labels) - labels.sum()) / len(labels)}")

# create balanced test and val sets
rus = RandomUnderSampler(
    sampling_strategy={0:n_class, 1:n_class}, random_state=random_seed)

test_x, test_y = rus.fit_resample(inputs, labels)
test_idx = rus.sample_indices_

print("\nResampled test data set:")
print(f"\t% spam: {100 * test_y.sum() / len(test_y)}")
print(f"\t% ham: {100 * (len(test_y) - test_y.sum()) / len(test_y)}")

# other indices
rest_idx = [i for i in range(inputs.shape[0]) if i not in test_idx]
rest_x, rest_y = inputs[rest_idx], labels[rest_idx]

# create valid data
valid_x, valid_y = rus.fit_resample(rest_x, rest_y)
valid_idx = rus.sample_indices_

print("\nResampled valid data set:")
print(f"\t% spam: {100 * valid_y.sum() / len(valid_y)}")
print(f"\t% ham: {100 * (len(valid_y) - valid_y.sum()) / len(valid_y)}")

# create train data
train_idx = [i for i in range(rest_x.shape[0]) if i not in valid_idx]
train_x, train_y = rest_x[train_idx], rest_y[train_idx]

print("\nTrain data set:")
print(f"\t% spam: {100 * train_y.sum() / len(train_y)}")
print(f"\t% ham: {100 * (len(train_y) - train_y.sum()) / len(train_y)}")


# resample train data with near-miss to increase learning distances
# create bag-of-word representation
countvec = CountVectorizer()
train_bow = countvec.fit_transform(train_x.reshape(-1).tolist())

# create near-miss data set
nm = NearMiss()
x_res, y_res = nm.fit_resample(train_bow, train_y)
train_idx = nm.sample_indices_
train_x, train_y = train_x[train_idx], train_y[train_idx]

print("\nNear-miss train data set:")
print(f"\t% spam: {100 * train_y.sum() / len(train_y)}")
print(f"\t% ham: {100 * (len(train_y) - train_y.sum()) / len(train_y)}")


#%% MODEL CREATION - BERT
# need tokenizer, encoder, classification head

# explore bert tokenizer
vocab_file = os.path.join('utils', 'vocab.txt')
do_lower_case = True

tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(vocab_file=vocab_file,
                                                      lower_case=do_lower_case)

# Token ID example
tokens = tf.reshape(tokenizer(["She sells seashells by the seashore"]), [-1])
print(f"Tokens IDs generated by BERT: {tokens}".format(tokens))
idx = [tokenizer._vocab[tid] for tid in tokens]
print(f"Tokens generated by BERT: {idx}")

# Special token ID example
special_tokens = ['[CLS]', '[SEP]', '[MASK]', '[PAD]']
idx = [tokenizer._vocab.index(tok) for tok in special_tokens]
for t, i in zip(special_tokens, idx):
    print(f'Token: {t} has ID: {i}')
    
    
#%% HELPER FUNCTIONS
def encode_sentence(s):
    # tokenize a sentence for bert to understand
    tokens = list(
        tf.reshape(tokenizer(["[CLS] " + s + " [SEP]"]), -1))
    return tokens

print(encode_sentence("I like ice cream"))

# create input_word_idx, input_mask, input_type_idx for BERT
def get_bert_inputs(tokenizer, docs, max_seq_len=None):
    
    packer = tfm.nlp.layers.BertPackInputs(seq_length=max_seq_len,
                                           special_tokens_dict=tokenizer.get_special_tokens_dict())
    
    packed = packer(tokenizer(docs))
    
    packed_numpy = dict(
        [(k, v.numpy()) for k, v in packed.items()])
    
    return packed_numpy


train_inputs = get_bert_inputs(tokenizer, train_x, max_seq_len=max_seq_length)
valid_inputs = get_bert_inputs(tokenizer, valid_x, max_seq_len=max_seq_length)
test_inputs = get_bert_inputs(tokenizer, test_x, max_seq_len=max_seq_length)

# shuffle training data
train_idx = np.random.permutation(len(train_inputs["input_word_ids"]))
train_inputs = dict(
    [(k, v[train_idx]) for k, v in train_inputs.items()])
train_y = train_y[train_idx]


#%% IMPORT BERT MODEL

# define inputs
input_word_ids = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
input_mask = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
input_type_ids = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

# download BERT encoder
class create_bert_encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        
        input_word_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
        
        self.inputs = {"input_word_ids": input_word_ids,
                       "input_mask": input_mask,
                       "input_type_ids": input_type_ids}
        
        self.layer = hub.KerasLayer(hub_bert_url, trainable=True)
        
    def call(self, x):
        return self.layer(x)
bert_layer = create_bert_encoder()


# define outputs
output = bert_layer(inputs)

outputs = {"sequence_output": output["sequence_output"],
           "pooled_output": output["pooled_output"]}

# final encoder model
hub_encoder = tf.keras.models.Model(inputs=inputs, outputs=outputs)

#%%
# class create_bert_classifier(tf.keras.layers.Layer):
    # def __init__(self, network)
bert_classifier = tfm.nlp.models.BertClassifier(network=hub_encoder,
                                                num_classes=num_classes)
