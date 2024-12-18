import tensorflow as tf
import tensorflow.keras.layers as layers

class FCLayer(layers.Layer):
    
    def __init__(self, d1, d2):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
    
    def build(self, input_shape):
        self.W1 = self.add_weight(
            shape=(input_shape[-1], self.d1),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32")
        
        self.b1 = self.add_weight(
            shape=(self.d1,),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32")
        
        self.W2 = self.add_weight(
            shape=(self.d1, self.d2),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32")
        
        self.b2 = self.add_weight(
            shape=(self.d2,),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32")
    
    def call(self, x):
        fcl1 = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        fcl2 = tf.matmul(fcl1, self.W2) + self.b2
        return fcl2