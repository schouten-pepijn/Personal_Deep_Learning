import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


class SelfAttentionLayer(layers.Layer):
    
    def __init__(self, d):
        super().__init__()
        self.d = d
    
    def build(self, input_shape):
        self.Wq = self.add_weight(
            shape=(input_shape[-1], self.d),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32"
            )
        
        self.Wk = self.add_weight(
            shape=(input_shape[-1], self.d),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32"
            )
        
        self.Wv = self.add_weight(
            shape=(input_shape[-1], self.d),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32"
            )
    
    def call(self, q_x, k_x, v_x, mask=None):
        q = tf.matmul(q_x, self.Wq)
        k = tf.matmul(k_x, self.Wk)
        v = tf.matmul(v_x, self.Wv)
        
        p = tf.nn.softmax(
            tf.matmul(q, k, transpose_b=True) / math.sqrt(self.d))
        
        h = tf.matmul(p, v)
        return h, p

"""
layer = SelfAttentionLayer(d=512)

n_seq = 7
x = tf.constant(np.random.normal(size=(1, n_seq, 512)))
h, p = layer(x, x, x)
print(p.shape)
print(h.shape)

#%% MULTIHEAD ATTENTION
multi_attn_head = [SelfAttentionLayer(64) for i in range(8)]
outputs = [head(x, x, x)[0] for head in multi_attn_head]
outputs = tf.concat(outputs, axis=-1)
print(outputs.shape)
"""