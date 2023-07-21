import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from train import ModelGeneration

def LoadModel(cfg, which='last'):
    path = 'model_path'
    model_names = glob.glob(os.path.join(cfg[path],'*.h*5'))
    model_names = [name for name in model_names if cfg['test_name'] in name]
    if isinstance(which, str):
        model_name = [name for name in model_names if (cfg['test_name']+'.h5' in name)]
        if not model_name:
            raise Exception('model does not exist')
        model_name = model_name[0]
    else:
        which = f'{which:03d}'
        model_name = [name for name in model_names if which in name]
        if not model_name:
            raise Exception('model does not exist')
        model_name = model_name[0]
    model, _ = ModelGeneration(cfg)
    model.load_weights(os.path.join(cfg[path], model_name))
    return model

def SavePredictions(model, dataset, cfg, save_dir):
    images = dataset['images']
    masks = dataset['masks']
    threshold = 0.5

    predictions = model.predict(images, verbose=1)
    predictions = (predictions > threshold)
    # Create a directory to save the predictions
    os.makedirs(save_dir, exist_ok=True)

    for i, (image, mask, preds) in enumerate(zip(images, masks, predictions)):
        # Rescale the image, mask, and prediction values
        image = (image*255).astype(np.uint8)
        mask = np.expand_dims(mask, axis=-1)
        mask = (mask  * 255).astype(np.uint8)
        preds = (preds * 255.0).astype(np.uint8)


        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        #fig.suptitle('Prediction for model of ' + experiment + ' from ' + pred_data + ' dataset', fontweight="bold", size=15)
        ax[0].imshow(image)
        ax[0].set_title('Original image')
        ax[0].axis('off')
    
        ax[1].imshow(mask.squeeze())
        ax[1].set_title('Ground truth mask')
        ax[1].axis('off')
    
        ax[2].imshow(preds.squeeze())
        ax[2].set_title('Predicted mask')
        ax[2].axis('off')
        
        # ''' Contour the predicted mask on the original image'''
        # ax[3].imshow(X[ix])
        # ax[3].contour(np.squeeze(preds[ix]))
        # ax[3].set_title('Smoke detection')
        # ax[3].axis('off')
        
        ''' Overlay the predicted mask on the original image'''
        overlaid_mask = np.ma.masked_where(preds == 0, preds)
        ax[3].imshow(image)
        ax[3].imshow(overlaid_mask, alpha=0.5)
        ax[3].set_title('Original Image and Predicted Mask Overlayed')
        ax[3].axis('off')

        # Save the figure
        plt.savefig(os.path.join(save_dir, f'prediction_{cfg["test_name"]}_{i:03d}.png'))
        plt.close()

def SavePredictions_new(model, dataset, cfg, save_dir):
    images = dataset['images']
    threshold = 0.5

    predictions = model.predict(images, verbose=1)
    predictions = (predictions > threshold)
    # Create a directory to save the predictions
    os.makedirs(save_dir, exist_ok=True)

    for i, (image, preds) in enumerate(zip(images, predictions)):
        # Rescale the image, mask, and prediction values
        image = (image*255).astype(np.uint8)
        preds = (preds * 255.0).astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        #fig.suptitle('Prediction for model of ' + experiment + ' from ' + pred_data + ' dataset', fontweight="bold", size=15)
        ax[0].imshow(image)
        ax[0].set_title('Original image')
        ax[0].axis('off')
    
        ax[1].imshow(preds.squeeze())
        ax[1].set_title('Predicted mask')
        ax[1].axis('off')
        
        # ''' Contour the predicted mask on the original image'''
        # ax[3].imshow(X[ix])
        # ax[3].contour(np.squeeze(preds[ix]))
        # ax[3].set_title('Smoke detection')
        # ax[3].axis('off')
        
        ''' Overlay the predicted mask on the original image'''
        overlaid_mask = np.ma.masked_where(preds == 0, preds)
        ax[2].imshow(image)
        ax[2].imshow(overlaid_mask, alpha=0.5)
        ax[2].set_title('Original Image and Predicted Mask Overlayed')
        ax[2].axis('off')

        # Save the figure
        plt.savefig(os.path.join(save_dir, f'prediction_{cfg["test_name"]}_{i:03d}.png'))
        plt.close()

def Save_only_predict(model, dataset, cfg, save_dir):
    images = dataset['images']
    threshold = 0.5

    predictions = model.predict(images, verbose=1)
    predictions = (predictions > threshold)
    # Create a directory to save the predictions
    os.makedirs(save_dir, exist_ok=True)

    for i, (image, preds) in enumerate(zip(images, predictions)):
        # Rescale the image, mask, and prediction values
        image = (image*255).astype(np.uint8)
        preds = (preds * 255.0).astype(np.uint8)

        #fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        #fig, ax = plt.subplots()
        #fig.suptitle('Prediction for model of ' + experiment + ' from ' + pred_data + ' dataset', fontweight="bold", size=15)

        '''Predicted mask'''
        plt.imshow(preds.squeeze())
        #ax.set_title('Predicted mask')
        plt.axis('off')
       # plt.margins(x=0)

        # Save the figure
        plt.savefig(os.path.join(save_dir, f'prediction_{cfg["test_name"]}_{i:03d}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
