import tensorflow as tf


def log_io(func):
    def wrapper(*args, **kwargs):
        print("args: ", args)
        print("kwargs: ", kwargs)
        out = func(*args, **kwargs)
        print("return: ", out, "\n")
    return wrapper
        

@log_io
def easy_math(x, y):
    return x * y * (x * y)

res = easy_math(2, 3)

@tf.function
def f1(x, y, z):
    return tf.math.add(tf.math.multiply(x, y), z)


x, y, z = 3, 4, 6
with tf.device("/CPU:0"):
    w = f1(x, y, z)
print(w)