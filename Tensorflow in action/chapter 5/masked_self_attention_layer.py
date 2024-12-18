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
        
        p = tf.matmul(q, k, transpose_b=True) / math.sqrt(self.d)
        p = tf.squeeze(p)
        
        if mask is not None:
            p += mask * -1e9
            
        p = tf.nn.softmax(p)
        h = tf.matmul(p, v)
        return h, p
    
"""
layer = SelfAttentionLayer(d=512)


mask = tf.linalg.band_part(tf.ones((7,7)), 0, -1)

print(mask.numpy())

n_seq = 7

# no mask
x = tf.constant(np.random.normal(size=(1, n_seq, 512)))
h, p = layer(x, x, x)
print(p.numpy())

# mask
h, p = layer(x, x, x, mask=mask)
print(p.numpy())
"""