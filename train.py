import os
import tensorflow as tf
import keras
import wandb
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from unet_model import unet_model, dice_coef, dice_loss, iou, TPR, TNR # import funtion of unet model
#from unet_model_original import unet_model, dice_coef, dice_loss, iou, TPR, TNR # import funtion of unet model

from wandb.keras import WandbCallback
from tensorflow.keras.metrics import Precision, Recall
from evaluation import LogImagesCallback
from tensorflow.keras.optimizers.schedules import CosineDecay
from wandb.keras import WandbMetricsLogger


def ModelGeneration(cfg, Dataset=None):
    keras.backend.clear_session()
    model = unet_model(cfg=cfg)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate'])
    metrics=['accuracy', Precision(), Recall(), dice_coef, iou, TPR, TNR] #, f1_score
    loss_metric = dice_loss
    #loss_metric = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    
    #Saves the best model and weights to use it later
    model_path = os.path.join(cfg['model_path'],cfg['test_name'])+"_{epoch:03d}.hdf5"
    

    measure = 'val_accuracy'; mode = 'max'

    callbacks = []
    if Dataset:
        every = cfg['save_every_epoch']*(Dataset['images'].shape[0])//cfg['batch_size']
        model_checkpoint_callback = ModelCheckpoint(model_path, save_freq=every, verbose=1, save_weights_only=True)
        callbacks.append(model_checkpoint_callback)
        callbacks.append(WandbCallback())
        callbacks.append(WandbMetricsLogger())
        
    
    if cfg['lr_scheduler']=='None':
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate'])
    else:
        LR_scheduler = CosineDecay(initial_learning_rate=cfg['learning_rate'], decay_steps=1000)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR_scheduler)

    metrics=['accuracy', Precision(), Recall(), dice_coef, iou, TPR, TNR] #, f1_score
    loss_metric = dice_loss
  #  loss_metric = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)

    return model, callbacks


def TrainModel(model, callbacks, TrainDataset, ValDataset, cfg):
    callbacks.append(LogImagesCallback(ValDataset))
    history = model.fit(TrainDataset['images'], TrainDataset['masks'],
                    batch_size=cfg['batch_size'],
                    epochs=cfg['num_epochs'],
                    validation_data=(ValDataset['images'], ValDataset['masks']),
                    shuffle=True,
                    callbacks=callbacks,
                    verbose=1)
    model.save(os.path.join(cfg['model_path']+cfg['test_name']+'.h5'))
   
    return model, history