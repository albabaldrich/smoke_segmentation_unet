'''
Original Unet model
'''
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
import numpy as np

def TPR(y_true, y_pred):
    TP = K.sum(y_true*y_pred)
    all_pos = K.sum(y_true)
    
    return TP/(all_pos + K.epsilon())

def TNR(y_true, y_pred):
    TN = K.sum((1-y_true)*(1-y_pred))
    all_neg = K.sum(1-y_true)
    
    return TN/(all_neg + K.epsilon())

def iou(y_true, y_pred, smooth=1):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    
    return (intersection + smooth) / (union + smooth)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    return (2. * intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


################################################################

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input, MaxPooling2D
from tensorflow.keras.models import Model


def conv_block(input, filters):
    x = Conv2D(filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


# Build the model
def unet_model(cfg):
    input_shape = (cfg['img_width'], cfg['img_height'], 3)
    inputs = Input(input_shape)

    # Contracting Path (Encoder)
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv_block(p4, 1024)

    # Expanding Path (Decoder)
    u6 = Conv2DTranspose(512, (2, 2), strides=2, padding='same')(c5)
    u6 = Concatenate()([u6, c4])
    u6 = conv_block(u6, 512)

    u7 = Conv2DTranspose(256, (2, 2), strides=2, padding='same')(u6)
    u7 = Concatenate()([u7, c3])
    u7 = conv_block(u7, 256)

    u8 = Conv2DTranspose(128, (2, 2), strides=2, padding='same')(u7)
    u8 = Concatenate()([u8, c2])
    u8 = conv_block(u8, 128)

    u9 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(u8)
    u9 = Concatenate()([u9, c1])
    u9 = conv_block(u9, 64)

    outputs = Conv2D(1, 1, activation='sigmoid', padding="same")(u9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
