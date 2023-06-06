'''
Unet with ResNet50 autoencoder

from tensorflow.keras.applications import ResNet50
# load the model
model = ResNet50()
# print the summary
model.summary()
'''


from keras import backend as K
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
import numpy as np

def TPR(y_true, y_pred):
#    TP = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
#    all_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    TP = K.sum(y_true*y_pred)
    all_pos = K.sum(y_true)
    
    return TP/(all_pos + K.epsilon())

def TNR(y_true, y_pred):
#    TN = K.sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
#    all_neg = K.sum(K.round(K.clip(1-y_true, 0, 1)))
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
#class DiceLoss(tf.keras.losses.Loss):
#  def __init__(self, num_classes):
      

#precision = Precision()
#recall = Recall()

def f1_score(y_true, y_pred):
    P = precision(y_true, y_pred)
    R = recall(y_true, y_pred)
    f1 = 2 * ((P * R) / (P + R + K.epsilon()))
    
    return f1

#EDGE_CROP = 2

################################################################
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import Dropout, GaussianNoise  #Regularization layers 
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNetV2

   
def conv_block(input, filters):
    x = Conv2D(filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


#Build the model
def unet_model(cfg):
    if cfg['pretrained_encoder'].lower() == 'resnet50':
        return unet_ResNet50(cfg)
    else:
        return unet_MobileNetV2(cfg)


def unet_ResNet50(cfg):
    #Input
    input_shape = (cfg['img_width'], cfg['img_height'], 3)
    inputs = Input(input_shape)
    
    '''Contraction part (encoder) - ResNet50 autoencoder'''
    encoder = ResNet50(include_top=False, weights='imagenet', input_tensor=  inputs)
    if cfg.get('transfer', 0):
        encoder.trainable = False
    
    '''Contractive path (encoder)'''
    c1 = encoder.get_layer('input_1').output
    c2 = encoder.get_layer('conv1_relu').output
    c3 = encoder.get_layer('conv2_block3_out').output
    c4 = encoder.get_layer('conv3_block4_out').output
    
    '''Bridge'''
    b = encoder.get_layer('conv4_block6_out').output
        
#    if cfg['dropout']>0:
#        layer = Dropout(cfg['dropout'])(layer)

    '''Expansive path (decoder)'''
    u6 = Conv2DTranspose(512, (2, 2), strides=2, padding='same') (b)
    u6 = Concatenate()([u6, c4])
    u6 = conv_block(u6, 512)
    
    u7 = Conv2DTranspose(256, (2, 2), strides=2, padding='same') (u6)
    u7 = Concatenate()([u7, c3])
    u7 = conv_block(u7, 256) 
    
    u8 = Conv2DTranspose(128, (2, 2), strides=2, padding='same') (u7)
    u8 = Concatenate()([u8, c2])
    u8 = conv_block(u8, 128)
    if cfg['dropout']>0:
        u8 = Dropout(cfg['dropout'])(u8)   

    u9 = Conv2DTranspose(64, (2, 2), strides=2, padding='same') (u8)
    u9 = Concatenate()([u9, c1])
    u9 = conv_block(u9, 64)
    # if cfg['dropout']>0:
    #     u9 = Dropout(cfg['dropout'])(u9)
    
    outputs = Conv2D(1, 1, activation='sigmoid', padding="same") (u9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model 


def unet_MobileNetV2(cfg):
    input_shape = (cfg['img_width'], cfg['img_height'], 3)
    inputs = Input(shape=input_shape, name="input_image")
    
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    if cfg.get('transfer', 0):
        encoder.trainable = False

    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    f = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        x = conv_block(x, f[-i])
        if (cfg['dropout']>0) and (i == len(skip_connection_names)-2):
            x = Dropout(cfg['dropout'])(x)               
        
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    return model