import numpy as np
import wandb
import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os


def wb_mask(bg_img, pred_mask, true_mask):
    labels =  {0: "clear", 1: "smoke"} 
    return wandb.Image((bg_img*255).astype(np.uint8), masks={
        "prediction" : {"mask_data" : (pred_mask*1).astype(np.uint8), "class_labels" : labels},
        "ground truth" : {"mask_data" : (true_mask*1).astype(np.uint8), "class_labels" : labels}})

    
def ShowPredictions(model, Dataset, title='Predictions', num=4):
    if type(Dataset) is dict:
        images = Dataset['images']
        masks = Dataset['masks']
    else:
        images = Dataset[0]
        masks = Dataset[1]
        
    predictions = model.predict(images, verbose=1)
    predictions = predictions.squeeze() > 0.5
    
    mask_list = []
    for i, (image, mask, mask_pred)  in enumerate(zip(images, masks, predictions)):
        if i==num:
            break
        mask_list.append(wb_mask(image, mask_pred, mask))
        
    # log all composite images to W&B
    wandb.log({title : mask_list})
    
    
def ShowTable(model, Dataset, title='Predictions'):
    predictions = model.predict(Dataset['images'], verbose=1)
    predictions = predictions.squeeze() > 0.5

    table = wandb.Table(columns=['Original Image', 'Original Mask', 'Predicted Mask'], allow_mixed_types = True)

    for (image, mask, mask_pred)  in zip(Dataset['images'],Dataset['masks'], predictions):
        table.add_data(
            wandb.Image((image*255).astype(np.uint8)),
            wandb.Image((mask*255).astype(np.uint8)),
            wandb.Image((mask_pred*255).astype(np.uint8))
        )

    wandb.log({title: table})



def ConfusionMatrix(model, dataset, cfg, save_dir):
    ''' Confusion matrix (validation)'''
    # Prediction of the validation dataset
    images = dataset['images']
    masks = dataset['masks']
    threshold = 0.5

    predictions = model.predict(images, verbose=1)>threshold

    cm= confusion_matrix(masks.flatten(), predictions.flatten())

    # Normalise the confusion matrix
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = ['Smoke (0)', "Non smoke (1)"]


    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cmn, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues', annot_kws={'fontsize': 15}, fmt = '0.4f')
    ax.set_title(('Confusion matrix of ' + cfg['test_name']), fontsize = 15)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    plt.savefig(os.path.join(save_dir + 'conf_matrix_' + cfg['test_name'] + '.png'))

    

#  callback extension to log image masks
class LogImagesCallback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(LogImagesCallback, self).__init__()
        self.valdata = validation_data

    def on_epoch_end(self, epoch, logs=None):
        ShowPredictions(self.model, self.valdata)
