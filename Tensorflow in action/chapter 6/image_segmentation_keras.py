import os 
os.chdir("/Users/pepijnschouten/Desktop/Deep Learning Projects/Tensorflow in action/chapter 6")
import tarfile
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from functools import partial
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.ops as ops

#%% EXTRACT DATA
tar_dir = "VOCtrainval_11-May-2012.tar"

if not os.path.exists(os.path.join("data", "VOCdevkit")):
    with tarfile.open(os.path.join("data", tar_dir), 'r') as tar:
        tar.extractall("data")
else:
    print("The tar is already extracted")


#%% INSPECT AN IMAGE

folder_path = os.path.join("data", "VOCdevkit", "VOC2012", "JPEGImages")
file_path = os.listdir(folder_path)[0]
image_path = os.path.join(folder_path, file_path)
orig_image = Image.open(image_path)

print(f"The format of the data is {orig_image.format}")
print(f"This image is of size: {orig_image.size}")


#%% RESTORE ORIGINAL IMAGE FROM PALETTE
def rgb_image_from_palette(image):
    palette = image.get_palette()
    
    palette = np.array(palette).reshape(-1, 3)
    if isinstance(image, PngImageFile):
        h, w = image.height, image.weight
        image = np.array(image).reshape(-1)
    elif isinstance(image, np.ndarray):
        h, w = image.shape[0], image.shape[1]
        image = image.reshape(-1)
    
    rgb_image = np.zeros(shape=(image.shape[0], 3))
    rgb_image[(image != 0), :] = palette[image[(image != 0)], :]
    
    return rgb_image


#%% TENSORFLOW TF.DATA DATASET PIPELINE
random_seed = 87
def get_subset_filenames(orig_dir, seg_dir, subset_dir, subset):
    """get the filenames for a given subset (train/valid/test)"""
    
    if subset.startswith("train"):
        ser = pd.read_csv(
            os.path.join(subset_dir, "train.txt"),
            index_col=None, header=None).squeeze("columns").tolist()
    
    elif subset.startswith("val") or subset.startswith("test"):
        random.seed(random_seed)
        
        ser = pd.read_cev(
            os.path.join(subset_dir, "val.txt"),
            index_col=None, header=None, squeeze=True).tolist()
        
        random.shuffle(ser)
        
        if subset.startswith("val"):
            ser = ser[:len(ser)//2]
        else:
            ser = ser[len(ser)//2:]
    else:
        raise NotImplementedError(f"subset={subset} is not valid")
        
    orig_filenames = [os.path.join(orig_dir, f+".jpg") for f in ser]
    seg_filenames = [os.path.join(seg_dir, f+".png") for f in ser]
    
    for o, a in zip(orig_filenames, seg_filenames):
        yield o, a
     

def load_image_func(image):
    img = np.array(Image.open(image))
    return img    

def get_subset_tf_dataset(
        subset_filename_gen_func, batch_size, epochs, input_size=(256, 256),
        output_size=None, resize_to_before_crop=None, augmentation=False,
        shuffle=False):
    
    filename_ds = tf.data.Dataset.from_generator(subset_filename_gen_func,
                                                 output_types=(tf.string, tf.string))
    
    image_ds = filename_ds.map(lambda x,y: (tf.image.decode_jpeg(tf.io.read_file(x)),
                                            tf.numpy_function(load_image_func, [y], [tf.uint8])
                                            )).cache()
    
    image_ds = image_ds.map(lambda x,y: (tf.cast(x, "float32")/255.0, y))
    
    def randomly_crop_or_resize(x, y):
        def rand_crop(x, y):
            x = tf.image.resize(x, resize_to_before_crop, method="bilinear")
            y = tf.cast(tf.image.resize(
                tf.transpose(y, [1,2,0]),
                resize_to_before_crop, method="nearest"
                ), "float32")
            
            offset_h = tf.random.uniform(
                [], 0, x.shape[0]-input_size[0], dtype="int32")
            offset_w = tf.random.uniform(
                [], 0, x.shape[1]-input_size[1], dtype="int32")
            
            x = tf.image.crop_to_bounding_box(image=x,
                                              offset_height=offset_h,
                                              offset_width=offset_w,
                                              target_height=input_size[0],
                                              target_width=input_size[1])
            y = tf.image.crop_to_bounding_box(image=y,
                                              offset_height=offset_h,
                                              offset_width=offset_w,
                                              target_height=input_size[0],
                                              target_width=input_size[1])
            
            return x, y
        
        def resize(x, y):
            x = tf.image.resize(x, input_size, method="bilinear")
            y = tf.cast(tf.image.resize(tf.transpose(y, [1,2,0]),
                                             input_size, method="nearest"), "float32")
            return x, y
        
        rand = tf.random.uniform([], 0.0, 1.0)
        
        if augmentation and \
            (input_size[0] < resize_to_before_crop[0] or \
             input_size[1] < resize_to_before_crop[1]):
                x, y = tf.cond(
                    rand < 0.5,
                    lambda: rand_crop(x, y),
                    lambda: resize(x, y))
        else:
            x, y = resize(x, y)
            
        return x, y
    
    
    def fix_shape(x, y, size):
        x.set_shape((size[0], size[1], 3))
        y.set_shape((size[0], size[1], 1))
        
        return x, y
    
    
    def randomly_flip_horizontal(x, y):
        rand = tf.random.uniform([], 0.0, 1.0)
        
        def flip(x, y):
            return tf.image.flip_left_right(x), tf.image.flip_left_right(x)
        
        x, t = tf.cond(rand < 0.5, lambda: flip(x, y), lambda: (x, y))
        
        return x, y
    
    image_ds = image_ds.map(lambda x, y: randomly_crop_or_resize(x, y))
    image_ds = image_ds.map(lambda x, y: fix_shape(x, y, size=input_size))
    
    if augmentation:
        image_ds = image_ds.map(lambda x, y: randomly_flip_horizontal(x, y))
        image_ds = image_ds.map(lambda x, y: (tf.image.random_hue(x, 0.1), y))
        image_ds = image_ds.map(lambda x, y: (tf.image.random_brightness(x, 0.1), y))
        image_ds = image_ds.map(lambda x, y: (tf.image.random_contrast(x, 0.8, 1.2), y))
    
    if output_size:
        image_ds = image_ds.map(lambda x, y: (x, tf.image.resize(y, output_size, method="nearest")))
    
    if shuffle:
        image_ds = image_ds.shuffle(buffer_size=batch_size*5)
    
    image_ds = image_ds.batch(batch_size).repeat(epochs)
    image_ds = image_ds.prefetch(tf.data.experimental.AUTOTUNE)
    image_ds = image_ds.map(lambda x, y: (x, tf.squeeze(y)))
    
    return image_ds
        

orig_dir = os.path.join("data", "VOCdevkit", "VOC2012", "JPEGImages")
seg_dir = os.path.join("data", "VOCdevkit", "VOC2012", "SegmentationClass")
subset_dir = os.path.join("data", "VOCdevkit", "VOC2012", "ImageSets", "Segmentation")

partial_subset_fn = partial(
    get_subset_filenames, orig_dir=orig_dir, seg_dir=seg_dir, subset_dir=subset_dir)

train_subset_fn = partial(partial_subset_fn, subset="train")
val_subset_fn = partial(partial_subset_fn, subset="val")
test_subset_fn = partial(partial_subset_fn, subset="test")

input_size = (384, 384)

batch_size = 32
epochs = 10
tr_image_ds = get_subset_tf_dataset(train_subset_fn, batch_size, epochs,
                                    input_size=input_size,
                                    resize_to_before_crop=(444, 444),
                                    augmentation=True, shuffle=True)

val_image_ds = get_subset_tf_dataset(val_subset_fn, batch_size, epochs,
                                     input_size=input_size,
                                     shuffle=False)

test_image_ds = get_subset_tf_dataset(test_subset_fn, batch_size, 1,
                                      input_size=input_size,
                                      shuffle=False)


#%% MODEL DEEPLAB v3 FUNCTINAL API
def block_level3(inp, filters, kernel_size, rate, block_id,
                 convlayer_id, activation=True):
    conv5_block_conv_out = layers.Conv2D(
        filters, kernel_size, dilation_rate=rate, padding='same',
        name=f"conv5_block{block_id}_{convlayer_id}_conv")(inp)
    
    conv5_block_bn_out = layers.BatchNormalization(
        name=f"conv5_block{block_id}_{convlayer_id}_bn")(conv5_block_conv_out)

    if activation:
        conv5_block_relu_out = layers.Activation(
            "relu", name=f"conv5_block{block_id}_{convlayer_id}_relu")(conv5_block_bn_out)

        return conv5_block_relu_out
    else:
        return conv5_block_bn_out
    
    
def block_level2(inp, rate, block_id):
    block_1_out = block_level3(inp, 512, (1,1), rate, block_id, 10)
    block_2_out = block_level3(block_1_out, 512, (3,3), rate, block_id, 20)
    block_3_out = block_level3(
        block_2_out, 2048, (1,1), rate, block_id, 30, activation=False)
    return block_3_out
    

def resnet_block(inp, rate):
    block0_out = block_level3(
        inp, 2048, (1,1), 1, block_id=1, convlayer_id=0, activation=False)
    
    block1_out = block_level2(inp, 2, block_id=1)
    block1_add = layers.Add(
        name=f"conv5_block{1}_add")([block0_out, block1_out])
    block1_relu = layers.Activation(
        "relu", name=f"conv5_block{1}_relu")(block1_add)
 
    block2_out = block_level2(block1_relu, 2, block_id=2) # no relu
    block2_add = layers.Add(name=f"conv5_block{2}_add")([block1_add, block2_out])
    block2_relu = layers.Activation(
        "relu", name=f"conv5_block{2}_relu")(block2_add)

    block3_out = block_level2 (block2_relu, 2, block_id=3)
    block3_add = layers.Add(
        name=f"conv5_block{3}_add")([block2_add, block3_out])
    block3_relu = layers.Activation(
        "relu", name=f"conv5_block{3}_relu")(block3_add)
    return block3_relu


def atrous_spatial_pyramid_pooling(inp):
    """ Defining the ASPP (Atrous spatial pyramid pooling) module """
    # Part A: 1x1 and atrous convolutions
    outa_1_conv = block_level3(
        inp, 256, (1,1), 1, '_aspp_a', 1, activation='relu')
    
    outa_2_conv = block_level3(
        inp, 256, (3,3), 6, '_aspp_a', 2, activation='relu')

    outa_3_conv = block_level3(
        inp, 256, (3,3), 12, '_aspp_a', 3, activation='relu')

    outa_4_conv = block_level3(
        inp, 256, (3,3), 18, '_aspp_a', 4, activation='relu')
    
    # Part B global pooling
    outb_1_avg = layers.Lambda(
        lambda x: ops.mean(x, axis=[1,2], keepdims=True))(inp)
    outb_1_conv = block_level3(
        outb_1_avg, 256, (1,1), 1, "_aspp_b", 1, activation="relu")
    outb_1_up = layers.UpSampling2D((12,12),
                                    interpolation="bilinear")(outb_1_conv)
    out_aspp = layers.Concatenate()([outa_1_conv, outa_2_conv, outa_3_conv,
                                     outa_4_conv, outb_1_up])
    
    return out_aspp


inp = layers.Input(shape=input_size+(3,))

resnet50 = tf.keras.applications.ResNet50(
    include_top=False, input_tensor=inp, pooling=None)

for layer in resnet50.layers:
    if layer.name == "conv5_block1_1_conv":
        break
out = layer.output 

print(out.name)

resnet50_up_to_conv4 = models.Model(resnet50.input, out)

resnet_block4_out = resnet_block(resnet50_up_to_conv4.output, 2)

out_aspp = atrous_spatial_pyramid_pooling(resnet_block4_out)
out = layers.Conv2D(21, (1,1), padding="same")(out_aspp)
final_out = layers.UpSampling2D((16,16), interpolation="bilinear")(out)

deeplabv3 = models.Model(resnet50_up_to_conv4.input, final_out)

# copy resnet conv5 weights to out resnetblock
w_dict = {}
for l in ["conv5_block1_0_conv", "conv5_block1_0_bn",
          "conv5_block1_1_conv", "conv5_block1_1_bn",
          "conv5_block1_2_conv", "conv5_block1_2_bn",
          "conv5_block1_3_conv", "conv5_block1_3_bn"]:
    w_dict[l] = resnet50.get_layer(l).get_weights()
    

#%% COMPILING THE MODEL
def get_label_weight(y_true, y_pred, num_classes):
    weights = tf.reduce_mean(tf.one_hot(y_true, num_classes), axis=[1,2])
    
    tot = tf.reduce_sum(weights, axis=-1, keepdims=True)
    
    weights = (tot - weights) / tot
    
    y_true = tf.reshape(y_true, [-1, y_pred.shape[1]*y_pred.shape[2]])
    
    y_weights = tf.gather(params=weights, indices=y_true, batch_dim=1)
    y_weights= tf.reshape(y_weights, [-1])


def ce_weighted_from_logits(num_classes):
    def criterion(y_true, y_pred):
        valid_mask = tf.cast(
            tf.reshape((y_true <= num_classes -1), [-1,1]), "int32")
        
        y_true = tf.cast(y_true, "int32")
        y_true.set_shape((None, y_pred.shape[1], y_pred.shape[2]))
        
        y_weights = get_label_weight(y_true, y_pred, num_classes)
        
        y_pred_unwrap = tf.reshape(y_pred, [-1, num_classes])
        y_true_unwrap = tf.reshape(y_true, [-1])
        
        output = tf.reduce_mean(
            y_weights * tf.nn.sparse_softmax_cross_entropy_with_logits(
                y_true_unwrap*tf.squeeze(valid_mask),
                y_pred_unwrap*tf.cast(valid_mask, "float32")))
        
        return output

    return criterion


def dice_loss_from_logits(num_classes):
    def criterion(y_true, y_pred):
        smooth = 1.
        
        y_true = tf.cast(y_true, "int32")
        y_true.set_shape((None, y_pred.shape[1], y_pred.shape[2]))
        
        y_weights = get_label_weight(y_true, y_pred, num_classes)
        
        y_pred = tf.nn.softmax(y_pred)
        
        y_true_unwrap = tf.reshape(y_true, [-1])
        y_true_unwrap = tf.cast(
            tf.one_hot(tf.cast(y_true_unwrap, "int32"), num_classes), "float32")
        y_pred_unwrap = tf.reshape(y_pred, [-1, num_classes])
        
        intersection = tf.reduce_sum(y_true_unwrap*y_pred_unwrap*y_weights)
        union = tf.reduce_sum((y_true_unwrap+y_pred_unwrap)*y_weights)
        
        score = (2.*intersection+smooth) / (union + smooth)
        loss = 1 - score
        
        return loss
    
    return criterion


def ce_dice_loss_from_logits(num_classes):
    def criterion(y_true, y_pred):
        loss = ce_weighted_from_logits(num_classes)(
            tf.cast(y_true, "int32"), y_pred) + \
            dice_loss_from_logits(num_classes)(
                y_true, y_pred)
        
        return loss
    
    return criterion

# metrics

num_classes = 22
optimizer = keras.optimizers.Adam()
deeplabv3.compile(
    loss=ce_dice_loss_from_logits(num_classes),
    optimizer=optimizer,)

for k, w in w_dict.items():
    deeplabv3.get_layer(k).set_weights(w)
    

#%% TRAINING
if not os.path.exists("eval"):
    os.mkdir("eval")

csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join("eval", "1_pre_trained_deeplabv3.log"))


deeplabv3.fit(
    x=tr_image_ds,
    validation_data=val_image_ds,
    epochs=2,
    )
