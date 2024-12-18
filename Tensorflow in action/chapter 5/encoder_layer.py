import os
os.chdir("/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow in action/chapter 5")
import tensorflow as tf
import tensorflow.keras.layers as layers
import self_attention_layer as sal
import linear_layer as ll
import numpy as np

class EncoderLayer(layers.Layer):
    
    def __init__(self, d, n_heads):
        super().__init__()
        
        self.d = d
        assert d % n_heads == 0, "dimension and number of heads incorrect"
        self.d_head = int(d / n_heads)
        self.n_heads = n_heads
        self.attn_heads = [sal.SelfAttentionLayer(self.d_head)
                           for i in range(self.n_heads)]
        self.fc_layer = ll.FCLayer(2048, self.d)
        
    def call(self, x):
        def compute_multihead_output(x):
            outputs = [head(x, x, x)[0] for head in self.attn_heads]
            return tf.concat(outputs, axis=-1)
        
        h1 = compute_multihead_output(x)
        y = self.fc_layer(h1)
        
        return y

"""
layer = EncoderLayer(d=512, n_heads=4)

x = tf.constant(np.random.normal(size=(1, 25, 512)))

y = layer(x)

print(y.shape)
"""

        