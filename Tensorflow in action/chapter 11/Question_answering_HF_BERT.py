from datasets import load_dataset
from transformers import DistilBertTokenizerFast
import tensorflow as tf
from functools import partial
from transformers import TFDistilBertForQuestionAnswering
from transformers import DistilBertConfig


#%% DATA IMPORT
# load squad v1 data set
dataset = load_dataset("squad")
print(dataset)

# print some answers
print(dataset["train"]["answers"][:5])


#%% DATA PROCESSING
def correct_indices_add_end_idx(answers, contexts):
    n_correct, n_fix = 0, 0
    fixed_answers = []
    
    # extract data
    for answer, context in zip(answers, contexts):
        gold_text = answer['text'][0]
        answer['text'] = gold_text
        start_idx = answer['answer_start'][0]
        answer['answer_start'] = start_idx
        if start_idx < 0 or len(gold_text.strip()) == 0:
            print(answer)
        end_idx = start_idx + len(gold_text)
        
        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
            n_correct += 1
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1
            n_fix += 1
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2
            n_fix += 1    
            
        fixed_answers.append(answer)
    

    print(f"\t{n_correct}/{len(answers)} examples had the correct answer indices")
    print(f"\t{n_fix}/{len(answers)} examples had the wrong answer indices")
    
    return fixed_answers, contexts


train_questions = dataset["train"]["question"]
train_answers, train_contexts = correct_indices_add_end_idx(
    dataset["train"]["answers"], dataset["train"]["context"])

test_questions = dataset["validation"]["question"]
test_answers, test_contexts = correct_indices_add_end_idx(
    dataset["validation"]["answers"], dataset["validation"]["context"])


#%% MODEL INPUT CREATION
# tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# example IDS
context = 'This is the context'
question = 'This is the question'
token_ids = tokenizer(context, question, return_tensors='tf')
print(token_ids)

# example tokes
print(tokenizer.convert_ids_to_tokens(token_ids['input_ids'].numpy()[0]))

# encoding training and test data
train_encodings = tokenizer(
    train_contexts, train_questions, truncation=True, padding=True,
    return_tensors='tf')
print(f"train_encodings.shape: {train_encodings['input_ids'].shape}")

test_encodings = tokenizer(
    test_contexts, test_questions, truncation=True, padding=True,
    return_tensors='tf')
print(f"test_encodings.shape: {test_encodings['input_ids'].shape}")

# convert char based indices to token based indices to indicate start of anws
def update_char_to_token_pos_inplace(encodings, answers):
    start_positions, end_positions, n_updates = [], [], 0
    
    for i in range(len(answers)):
        start_positions.append(
            encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(
            encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        
        if start_positions[-1] is None or end_positions[-1] is None:
            n_updates += 1
            
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length - 1
            
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length - 1
        
    print(f"{n_updates}/{len(answers)} had answers truncated")
    
    encodings.update({
        'start_positions': start_positions, 'end_positions': end_positions})
    
    
update_char_to_token_pos_inplace(train_encodings, train_answers)
update_char_to_token_pos_inplace(test_encodings, test_answers)
        

#%% TF.DATA PIPELINE
# create input and output tuples
def data_gen(input_ids, attention_mask, start_positions, end_positions):
    for inps, attn, start_pos, end_pos in zip(
            input_ids, attention_mask, start_positions, end_positions):
        yield (inps, attn), (start_pos, end_pos)
        
# create train and valid set
train_data_gen = partial(
    data_gen,
    input_ids=train_encodings['input_ids'],
    attention_mask=train_encodings['attention_mask'],
    start_positions=train_encodings['start_positions'],
    end_positions=train_encodings['end_positions'])

train_dataset = tf.data.Dataset.from_generator(
    train_data_gen, output_types=(('int32', 'int32'), ('int32', 'int32')))
train_dataset = train_dataset.shuffle(20000)

valid_dataset = train_dataset.take(10000)
valid_dataset = valid_dataset.batch(8)

train_dataset = train_dataset.skip(10000)
train_dataset = train_dataset.batch(8)

# create test set
test_data_gen = partial(data_gen,
    input_ids=test_encodings['input_ids'],
    attention_mask=test_encodings['attention_mask'],
    start_positions=test_encodings['start_positions'],
    end_positions=test_encodings['end_positions'])

test_dataset = tf.data.Dataset.from_generator(
    test_data_gen, output_types=(('int32', 'int32'), ('int32', 'int32')))
test_dataset = test_dataset.batch(8)

# print exmple train data
for data in train_dataset.take(1):
    print(data)


#%% MODEL CREATION
# create dict outputs
config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased", return_dict=True)

# create model
model = TFDistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased", config=config)


# wrapping the model in a keras model
def tf_wrap_model(model):
    input_ids = tf.keras.layers.Input(
        (None,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(
        (None,), dtype=tf.int32, name='attention_mask')
    out = model([input_ids, attention_mask])
    
    wrap_model = tf.keras.models.Model(
        inputs=[input_ids, attention_mask],
        outputs=[out.start_logits, out.end_logits])
    
    return wrap_model

model_v2 = tf_wrap_model(model)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)
acc = tf.keras.metrics.SparseCategoricalAccuracy()

model_v2.compile(loss=loss, optimizer=optimizer, metrics=[acc])


#%% TRAIN THE MODEL
model_v2.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=3)
