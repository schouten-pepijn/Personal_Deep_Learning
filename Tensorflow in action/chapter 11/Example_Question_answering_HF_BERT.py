import random
import numpy as np
import transformers
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from transformers import DistilBertConfig, TFDistilBertForQuestionAnswering
import tensorflow as tf
import time
from datasets import load_dataset
from functools import partial


def fix_random_seed(seed):
    """ Setting the random seed of various libraries """
    try:
        np.random.seed(seed)
    except NameError:
        print("Warning: Numpy is not imported. Setting the seed for Numpy failed.")
    try:
        tf.random.set_seed(seed)
    except NameError:
        print("Warning: TensorFlow is not imported. Setting the seed for TensorFlow failed.")
    try:
        random.seed(seed)
    except NameError:
        print("Warning: random module is not imported. Setting the seed for random failed.")
    try:
        transformers.trainer_utils.set_seed(seed)
    except NameError:
        print("Warning: transformers module is not imported. Setting the seed for transformers failed.")
        
# Fixing the random seed
random_seed=4321
fix_random_seed(random_seed)


#%% DATA IMPORTING
dataset = load_dataset("squad")

print(dataset)

print(dataset["train"]["answers"][:5])


#%% CORRECT INDICES
def correct_indices_add_end_idx(answers, contexts):
    """ Correct the answer index of the samples (if wrong) """
    
    # Track how many were correct and fixed
    n_correct, n_fix = 0, 0
    fixed_answers = []
    for answer, context in zip(answers, contexts):

        gold_text = answer['text'][0]
        answer['text'] = gold_text
        start_idx = answer['answer_start'][0]
        answer['answer_start'] = start_idx
        if start_idx <0 or len(gold_text.strip())==0:
            print(answer)
        end_idx = start_idx + len(gold_text)        
        
        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
            n_correct += 1
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
            n_fix += 1
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters
            n_fix +=1
        
        fixed_answers.append(answer)
        
    # Print how many samples were fixed
    print("\t{}/{} examples had the correct answer indices".format(n_correct, len(answers)))
    print("\t{}/{} examples had the wrong answer indices".format(n_fix, len(answers)))
    return fixed_answers, contexts

train_questions = dataset["train"]["question"]
print("Training data corrections")
train_answers, train_contexts = correct_indices_add_end_idx(
    dataset["train"]["answers"], dataset["train"]["context"]
)
test_questions = dataset["validation"]["question"]
print("\nValidation data correction")
test_answers, test_contexts = correct_indices_add_end_idx(
    dataset["validation"]["answers"], dataset["validation"]["context"]
)


#%% TOKENIZER
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# example
context = "This is the context"
question = "This is the question"

token_ids = tokenizer(context, question, return_tensors='tf')
print(token_ids)
print(tokenizer.convert_ids_to_tokens(token_ids['input_ids'].numpy()[0]))


# converting inputs to tokens
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True, return_tensors='tf')
print("train_encodings.shape: {}".format(train_encodings["input_ids"].shape))
# Encode test data
test_encodings = tokenizer(test_contexts, test_questions, truncation=True, padding=True, return_tensors='tf')
print("test_encodings.shape: {}".format(test_encodings["input_ids"].shape))


#%% DEALING WITH TRUNCATED DATA
def update_char_to_token_positions_inplace(encodings, answers):
    start_positions = []
    end_positions = []
    n_updates = 0
    # Go through all the answers
    for i in range(len(answers)):        
        
        # Get the token position for both start end char positions
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        
        if start_positions[-1] is None or end_positions[-1] is None:
            n_updates += 1
        # if start position is None, the answer passage has been truncated
        # In the guide, https://huggingface.co/transformers/custom_datasets.html#qa-squad
        # they set it to model_max_length, but this will result in NaN losses as the last
        # available label is model_max_length-1 (zero-indexed)
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length -1
            
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length -1
            
    print("{}/{} had answers truncated".format(n_updates, len(answers)))
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

update_char_to_token_positions_inplace(train_encodings, train_answers)
update_char_to_token_positions_inplace(test_encodings, test_answers)


#%% CREATE DATA PIPELINE
def data_gen(input_ids, attention_mask, start_positions, end_positions):
    """ Generator for data """
    for inps, attn, start_pos, end_pos in zip(input_ids, attention_mask, start_positions, end_positions):
        
        yield (inps, attn), (start_pos, end_pos)
        
print("Creating train data")

# Define the generator as a callable (not the generator it self)
train_data_gen = partial(data_gen,
    input_ids=train_encodings['input_ids'], attention_mask=train_encodings['attention_mask'],
    start_positions=train_encodings['start_positions'], end_positions=train_encodings['end_positions']
)

# Define the dataset
train_dataset = tf.data.Dataset.from_generator(
    train_data_gen, output_types=(('int32', 'int32'), ('int32', 'int32'))
)
# Shuffling the data
train_dataset = train_dataset.shuffle(1000)
print('\tDone')

batch_size = 8
# Valid set is taken as the first 10000 samples in the shuffled set
valid_dataset = train_dataset.take(10000)
valid_dataset = valid_dataset.batch(batch_size)

# Rest is kept as the training data
train_dataset = train_dataset.skip(10000)
train_dataset = train_dataset.batch(batch_size)

# Creating test data
print("Creating test data")

# Define the generator as a callable
test_data_gen = partial(data_gen,
    input_ids=test_encodings['input_ids'], attention_mask=test_encodings['attention_mask'],
    start_positions=test_encodings['start_positions'], end_positions=test_encodings['end_positions']
)
test_dataset = tf.data.Dataset.from_generator(
    test_data_gen, output_types=(('int32', 'int32'), ('int32', 'int32'))
)
test_dataset = test_dataset.batch(batch_size)
print("\tDone")


#%% DEFINING THE MODEL
config = DistilBertConfig.from_pretrained("distilbert-base-uncased", return_dict=False)
model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased", config=config)

# Code listing 13.8
def tf_wrap_model(model):
    """ Wraps the huggingface's model with in the Keras Functional API """
    
    # If this is not wrapped in a keras model by taking the correct tensors from
    # TFQuestionAnsweringModelOutput produced, you will get the following error
    # setting return_dict did not seem to work as it should
    
    # TypeError: The two structures don't have the same sequence type. 
    # Input structure has type <class 'tuple'>, while shallow structure has type 
    # <class 'transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput'>.
    
    # Define inputs
    input_ids = tf.keras.layers.Input([None,], dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input([None,], dtype=tf.int32, name="attention_mask")
    
    # Define the output (TFQuestionAnsweringModelOutput)
    out = model([input_ids, attention_mask])
    
    # Get the correct attributes in the produced object to generate an output tuple
    wrap_model = tf.keras.models.Model([input_ids, attention_mask],
                                       outputs=[out[0], out[1]])
    
    return wrap_model


# Define and compile the model

# Keras will assign a separate loss for each output and add them together. So we'll just use the standard CE loss
# instead of using the built-in model.compute_loss, which expects a dict of outputs and averages the two terms.
# Note that this means the loss will be 2x of when using TFTrainer since we're adding instead of averaging them.
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)

model_v2 = tf_wrap_model(model)
model_v2.compile(optimizer=optimizer, loss=loss, metrics=[acc])


#%% TRAIN THE MODEL
t1 = time.time()

model_v2.fit(
    train_dataset, 
    validation_data=valid_dataset,    
    epochs=3
)

t2 = time.time()

print("It took {} seconds to complete the training".format(t2-t1))