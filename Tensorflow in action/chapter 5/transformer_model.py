import os
os.chdir("/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow in action/chapter 5")
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import encoder_layer as el
import decoder_layer as dl


n_steps = 25
n_en_vocab = 300
n_de_vocab = 400
n_heads = 8
d = 512
mask = 1 - tf.linalg.band_part(
    tf.ones((n_steps, n_steps)), -1, 0)

en_inp = layers.Input(shape=(n_steps,))
en_emb = layers.Embedding(n_en_vocab, 512,
                          input_length=n_steps)(en_inp)
en_out1 = el.EncoderLayer(d, n_heads)(en_emb)
en_out2 = el.EncoderLayer(d, n_heads)(en_out1)

de_inp = layers.Input(shape=(n_steps,))
de_emb = layers.Embedding(n_de_vocab, 512,
                          input_length=n_steps)(de_inp)

de_out1 = dl.DecoderLayer(d, n_heads)(de_emb, en_out2,
                                      mask
                                      )
de_out2 = dl.DecoderLayer(d, n_heads)(de_out1, en_out2,
                                      mask
                                      )
de_pred = layers.Dense(n_de_vocab, activation="softmax")(de_out2)

transformer = models.Model(
    inputs=[en_inp, de_inp],
    outputs=de_pred,
    name="MinTransformer")


transformer.summary()

# criterion = keras.losses.CategoricalCrossentropy()
# optimizer = keras.optimizer.Adam()
# metrics = keras.metrics.Accuracy()
# transformer.compile(loss=criterion,
#                     optimizer=optimizer,
#                     metrics=[metrics])