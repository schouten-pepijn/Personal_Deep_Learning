import os
os.chdir("/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow in action/chapter 5")
import tensorflow as tf
import tensorflow.keras.layers as layers
import masked_self_attention_layer as msal
import self_attention_layer as sal
import linear_layer as ll
import numpy as np


class DecoderLayer(layers.Layer):
    
    def __init__(self, d, n_heads):
        super().__init__()
        
        self.d = d
        assert d % n_heads == 0, "dimension and number of heads incorrect"
        self.d_head = int(d / n_heads)
        self.n_heads = n_heads
        self.dec_attn_heads = [msal.SelfAttentionLayer(self.d_head)
                               for i in range(self.n_heads)]
        self.attn_heads = [sal.SelfAttentionLayer(self.d_head)
                               for i in range(self.n_heads)]
        self.fc_layer = ll.FCLayer(2048, self.d)
        
    def call(self, de_x, en_x, mask=None):
        def compute_multihead_output(de_x, en_x, attn_heads, mask=None):
            outputs = [head(en_x, en_x, de_x, mask)[0]
                       for head in attn_heads]
            return tf.concat(outputs, axis=-1)
        
        h1 = compute_multihead_output(de_x, en_x,
                                      self.dec_attn_heads, mask)
        h2 = compute_multihead_output(h1, en_x,
                                      self.attn_heads)

        y = self.fc_layer(h2)
        
        return y
    

"""
mask = 1 - tf.linalg.band_part(tf.ones((25, 25)), -1, 0)

de_emb = layers.Embedding(400, 512,
                          input_length=25)
# print(de_emb(tf.ones(shape=(1, 25))).shape)

layer = DecoderLayer(512, 8)

en_x = tf.constant(np.random.normal(size=(1, 25, 512)))

y = layer(de_emb(tf.ones(shape=(25,))), en_x, mask)

print(y.shape)
"""
