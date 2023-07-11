# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:58:27 2022

@author: Abaldrich

Metrics with dice coefficient (DSC)
"""

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


# def ConfusionMatrix(model, dataset):
#     class_names = ["Non-smoke", "Smoke"]
#     images = dataset['images']
#     masks = dataset['masks']
#     threshold = 0.5

#     predictions = model.predict(images, verbose=1)>threshold

#     wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None,
#                             y_true=masks.flatten(), preds=predictions.flatten(),
#                             class_names=class_names)})


def ConfusionMatrix(model, dataset, cfg, save_dir):
    ''' Confusion matrix (validation)'''
    # Prediction of the validation dataset
    images = dataset['images']
    masks = dataset['masks']
    threshold = 0.5

    predictions = model.predict(images, verbose=1)>threshold
    #y_pred_thresholded = y_pred >threshold


    # CONFUSION MATRIX
    # y_pred_flat = y_pred_thresholded.flatten()
    # y_pred_cm = y_pred_flat.astype(int)
    # val_masks_cm = (masks/255).flatten()

    cm= confusion_matrix(masks.flatten(), predictions.flatten())
    #cm = ConfusionMatrixDisplay.from_predictions(val_masks_cm, y_pred_cm)

    # Normalise the confusion matrix
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = ['Smoke (0)', "Non smoke (1)"]


    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cmn, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues', annot_kws={'fontsize': 15}, fmt = '0.4f')
    #sns.heatmap(cmn, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues', annot_kws={'fontsize': 15})
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
'''
# CONFUSION MATRIX (val)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred_flat = y_pred_thresholded.flatten()
y_pred_cm = y_pred_flat.astype(int)
val_masks_cm = (val_masks/255).flatten()

cm= confusion_matrix(val_masks_cm, y_pred_cm)
#cm = ConfusionMatrixDisplay.from_predictions(val_masks_cm, y_pred_cm)

# Normalise the confusion matrix
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
labels = ['Smoke (0)', "Non smoke (1)"]

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cmn, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues', annot_kws={'fontsize': 15}, fmt = '0.4f')
#sns.heatmap(cmn, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues', annot_kws={'fontsize': 15})
ax.set_title(('Confusion matrix of ' + test_name + ' (validation)'), fontsize = 15)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

fig.savefig('C:/Users/alba.baldrich/MyDisk/CERTEC-TFM/2023 Alba Baldrich/02 - image segmentation/UNet/output/metrics/conf_matrix/' + 'conf_matrix_val_' + (test_name) + '.png')


# Info obtained from the confusion matrix

tp = cmn[0,0]
fp = cmn[0,1]
fn = cmn[1,0]
tn = cmn[1,1]

#True positive rate (TPR)
TPR = tp/(tp+fn)

#True negative rate (TNR)
TNR = tn/(tn+fp)

#False positive rate (FPR)
FPR = 1-TNR

#False negative rate (FNR)
FNR = 1-TPR


# IOU (val)

# y_pred=model.predict(val_images)
# y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_pred, y_pred_thresholded)
union = np.logical_or(y_pred, y_pred_thresholded)
val_iou_score = np.sum(intersection) / np.sum(union)
print("Validation IoU socre is: ", val_iou_score)




### TRAINING ###
# Prediction of the training dataset
y_pred_train=best_model.predict(train_images)
y_pred_thresholded_train = y_pred_train > 0.5

# CONFUSION MATRIX (train)
y_pred_flat_train = y_pred_thresholded_train.flatten()
y_pred_cm_train = y_pred_flat_train.astype(int)
train_masks_cm = (train_masks/255).flatten()

cm_train = confusion_matrix(train_masks_cm, y_pred_cm_train)


# Normalise the confusion matrix
cmn_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
labels = ['Smoke (0)', "Non smoke (1)"]

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cmn_train, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues', annot_kws={'fontsize': 15}, fmt = '0.4f')
#sns.heatmap(cmn, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues', annot_kws={'fontsize': 15})
ax.set_title(('Confusion matrix of ' + test_name + ' (training)'), fontsize = 15)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

fig.savefig('C:/Users/alba.baldrich/MyDisk/CERTEC-TFM/2023 Alba Baldrich/02 - image segmentation/UNet/output/metrics/conf_matrix/' + 'conf_matrix_train_' + (test_name) + '.png')


# Info obtained from the confusion matrix

tp_t = cmn_train[0,0]
fp_t = cmn_train[0,1]
fn_t = cmn_train[1,0]
tn_t = cmn_train[1,1]

#True positive rate (TPR)
TPR_t = tp_t/(tp_t+fn_t)

#True negative rate (TNR)
TNR_t = tn_t/(tn_t+fp_t)

#False positive rate (FPR)
FPR_t = 1-TNR_t

#False negative rate (FNR)
FNR_t = 1-TPR_t


# IOU (train)
intersection_train = np.logical_and(y_pred_train, y_pred_thresholded_train)
union_train = np.logical_or(y_pred_train, y_pred_thresholded_train)
train_iou_score = np.sum(intersection_train) / np.sum(union_train)
print("Training IoU socre is: ", train_iou_score)


best_epoch=best_epoch-1

data = {
  "Train": [train_acc[best_epoch], train_loss[best_epoch], train_dice[best_epoch], train_iou_score, TPR_t, TNR_t, FPR_t, FNR_t],
  "Validation": [val_acc[best_epoch], val_loss[best_epoch], val_dice[best_epoch], val_iou_score, TPR, TNR, FPR, FNR]
}


df = pd.DataFrame(data, index=['Accuracy', 'Loss', 'DICE', 'IOU', 'TPR','TNR', 'FRP', 'FNR'])
print(df)

fig = plt.figure(figsize = (6, 0.3))
# ax = fig.add_subplot(111)
plt.axis('off')
table_results = plt.table(cellText = df.values, rowLabels = df.index, 
         colLabels = df.columns, cellLoc='center')
table_results.scale(1,2)
plt.title('Results of ' + test_name + ' (Batch size = ' + str(batch_size) + ' and lr = ' + str(learning_rate) + ')')
file_name_table = ('C:/Users/alba.baldrich/MyDisk/CERTEC-TFM/2023 Alba Baldrich/02 - image segmentation/UNet/output/metrics/table_results/' + 'results_' + (test_name) + '.png')
plt.savefig(file_name_table, bbox_inches = 'tight')
plt.show()



## Acc (wtf)
t_a = ((tp_t+tn_t)/(tp_t+tn_t+fp_t+fn_t))
print('train_acc = ', t_a)
print('tp = ', tp_t)
print('tn = ', tn_t)
print('fp = ', fp_t)
print('fn = ', fn_t)

v_a = ((tp+tn)/(tp+tn+fp+fn))
print('val_acc = ', v_a)
print('tp = ', tp)
print('tn = ', tn)
print('fp = ', fp)
print('fn = ', fn)
#################################################################

# # 5. MODEL PREDICTION

# # PLOT OK
# test_image_path = 'C:/Users/alba.baldrich/Desktop/UNet/Dataset/smoke_dataset/test/images'
# test_mask_path = 'C:/Users/alba.baldrich/Desktop/UNet/Dataset/smoke_dataset/test/masks'

# #pred_data = 'train'
# #pred_data = 'validation'
# pred_data = 'test'
# threshold = 0.5

# #Function to plot the results
# def plot_sample(X, y, preds, X1):
#     # if ix is None:
#     #     ix = random.randint(0, len(X))
#     # has_mask = y[ix].max() > 0
#     ix = random.randint(0, len(X))

#     fig, ax = plt.subplots(1, 4, figsize=(20, 5))
#     fig.suptitle('Prediction for model of ' + test_name, fontweight="bold", size=15)
#     ax[0].imshow(X[ix])
#     ax[0].set_title('Original image')

#     ax[1].imshow(y[ix].squeeze())
#     ax[1].set_title('Ground truth mask')

#     ax[2].imshow(preds[ix].squeeze())
#     ax[2].set_title('Predicted mask')
    
#     ax[3].imshow(X[ix])
#     ax[3].contour(np.squeeze(preds[ix]))
#     ax[3].set_title('Smoke detection')
    
#     fig.savefig('C:/Users/alba.baldrich/Desktop/UNet/model_comparisions/LR_batch_evolution/predictions/' + (test_name) + '_random_pred.png')


# if pred_data == 'train':
#     X_train = train_images
#     y_train = train_masks
#     preds_train = model.predict(X_train, verbose=1)
#     preds_train_t = (preds_train > 0.5).astype(np.uint8)
#     plot_sample(X_train, y_train, preds_train, preds_train_t)
#     img = train_images[:,:,:IMG_CHANNELS]
# elif pred_data == 'validation':
#     X_valid = val_images
#     y_valid = val_masks
#     preds_val = model.predict(X_valid, verbose=1)
#     preds_val_t = (preds_val > 0.5).astype(np.uint8)
#     plot_sample(X_valid, y_valid, preds_val, preds_val_t)
#     img = val_images[:,:,:IMG_CHANNELS]
# elif pred_data == 'test':
#     test_img_dataset = []
#     test_msk_dataset = []  # 0=smoke; 1=smoke

#     # Read images and masks dataset and resize them
#     # Read images and masks dataset
#     for i in glob.glob(test_image_path+'/*.jpg'):
#         test_image = cv2.imread(i)
#         test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
#         test_image = cv2.resize(test_image, dsize=(img_width, img_height),
#                             interpolation=cv2.INTER_NEAREST)
#         test_image = cv2.normalize(test_image, test_image, 0, 255, cv2.NORM_MINMAX)
#         test_img_dataset.append(test_image)

#     for i in glob.glob(test_mask_path+'/*.png'):
#         test_mask = cv2.imread(i)
#         test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)
#         test_mask = cv2.resize(test_mask, dsize=(img_width, img_height),
#                           interpolation=cv2.INTER_NEAREST)
#         test_mask = cv2.normalize(test_mask, test_mask, 0, 255, cv2.NORM_MINMAX)
#         test_msk_dataset.append(test_mask)

#     # convert the list of arrays to a single numpy array
#     test_img_dataset = np.stack(test_img_dataset)
#     test_msk_dataset = np.stack(test_msk_dataset)
    
#     X_test = test_img_dataset
#     y_test = test_msk_dataset

#     preds_test = model.predict(X_test, verbose=1)
#     preds_test = ((preds_test > threshold) * 255).astype(np.uint8)
#     preds_test_t = ((preds_test > threshold) * 255)/255.
#     preds_test_t = preds_test_t.astype(np.uint8)
#     plot_sample(X_test, y_test, preds_test, preds_test_t)


#################################################


# prova_image_path = 'C:/Users/alba.baldrich/Desktop/UNet/Dataset/dataset_only_new/prova/images'
# prova_mask_path = 'C:/Users/alba.baldrich/Desktop/UNet/Dataset/dataset_only_new/prova/masks'

# prova_img_dataset = []
# prova_msk_dataset = []  # 0=smoke; 1=smoke


# #Function to plot the results
# def plot_sample(X, y, preds, X1):
#     ix = 0
#     while (ix < 2):
#         ix = ix
#         fig, ax = plt.subplots(1, 4, figsize=(20, 5))
#         fig.suptitle('Prediction for model of ' + test_name, fontweight="bold", size=15)
#         ax[0].imshow(X[ix])
#         ax[0].set_title('Original image')
    
#         ax[1].imshow(y[ix].squeeze())
#         ax[1].set_title('Ground truth mask')
    
#         ax[2].imshow(preds[ix].squeeze())
#         ax[2].set_title('Predicted mask')
        
#         ax[3].imshow(X[ix])
#         ax[3].contour(np.squeeze(preds[ix]))
#         ax[3].set_title('Smoke detection')
            
#         if ix == 0:
#             if type_pred == 'grayscale':     
#                 fig.savefig('C:/Users/alba.baldrich/Desktop/UNet/model_comparisions/LR_batch_evolution/predictions/' + (test_name) + '.png')
#             elif type_pred == 'binary':
#                 fig.savefig('C:/Users/alba.baldrich/Desktop/UNet/model_comparisions/LR_batch_evolution/predictions/' + (test_name) + '_binary.png')
#         elif ix == 1:
#             if type_pred == 'grayscale': 
#                 fig.savefig('C:/Users/alba.baldrich/Desktop/UNet/model_comparisions/LR_batch_evolution/predictions/' + (test_name) + '_.png')
#             elif type_pred == 'binary':
#                 fig.savefig('C:/Users/alba.baldrich/Desktop/UNet/model_comparisions/LR_batch_evolution/predictions/' + (test_name) + '_binary_.png')
#         ix += 1

# # Read images and masks dataset and resize them
# # Read images and masks dataset
# for i in glob.glob(prova_image_path+'/*.jpg'):
#     test_image = cv2.imread(i)
#     test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
#     test_image = cv2.resize(test_image, dsize=(img_width, img_height),
#                         interpolation=cv2.INTER_NEAREST)
#     test_image = cv2.normalize(test_image, test_image, 0, 255, cv2.NORM_MINMAX)
#     prova_img_dataset.append(test_image)

# for i in glob.glob(prova_mask_path+'/*.png'):
#     test_mask = cv2.imread(i)
#     test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)
#     test_mask = cv2.resize(test_mask, dsize=(img_width, img_height),
#                       interpolation=cv2.INTER_NEAREST)
#     test_mask = cv2.normalize(test_mask, test_mask, 0, 255, cv2.NORM_MINMAX)
#     prova_msk_dataset.append(test_mask)

# # convert the list of arrays to a single numpy array
# prova_img_dataset = np.stack(prova_img_dataset)
# prova_msk_dataset = np.stack(prova_msk_dataset)


# type_pred = 'grayscale'

# X_test = prova_img_dataset
# y_test = prova_msk_dataset
# preds_test = model.predict(X_test, verbose=1)
# preds_test_t = (preds_test > 0.5).astype(np.uint8)
# plot_sample(X_test, y_test, preds_test, preds_test_t)
# plot_sample(X_test, y_test, preds_test, preds_test_t)



# type_pred = 'binary'

# X_test = prova_img_dataset
# y_test = prova_msk_dataset

# threshold = 0.5
# preds_test = model.predict(X_test, verbose=1)
# preds_test = ((preds_test > threshold) * 255).astype(np.uint8)
# preds_test_t = ((preds_test > threshold) * 255)/255.
# preds_test_t = preds_test_t.astype(np.uint8)


# plot_sample(X_test, y_test, preds_test, preds_test_t)



# #################################################
# # REPORT PHOTOS

# prova_image_path = 'C:/Users/alba.baldrich/Desktop/UNet/Dataset/dataset_only_new/prova/images'
# prova_mask_path = 'C:/Users/alba.baldrich/Desktop/UNet/Dataset/dataset_only_new/prova/masks'

# prova_img_dataset = []
# prova_msk_dataset = []  # 0=smoke; 1=smoke


# #Function to plot the results
# def plot_sample(X, y, preds, X1):
#     ix = 0
#     while (ix < 2):
#         ix = ix
#         fig, ax = plt.subplots(1, 4, figsize=(20, 5))
#         fig.suptitle('Prediction for model of ' + test_name, fontweight="bold", size=15)
#         ax[0].imshow(X[ix])
#         ax[0].set_title('Original image')
    
#         ax[1].imshow(y[ix].squeeze())
#         ax[1].set_title('Ground truth mask')
        
        
#         ax[2].imshow(preds[ix].squeeze())
#         ax[2].set_title('Predicted mask')
        
#         ax[3].imshow(X[ix])
#         ax[3].contour(np.squeeze(preds[ix]))
#         ax[3].set_title('Smoke detection')
            
#         if ix == 0:
#                 fig.savefig('C:/Users/alba.baldrich/Desktop/UNet/model_comparisions/LR_batch_evolution/predictions/' + (test_name) + '_binary.png')
#         elif ix == 1:
#             fig.savefig('C:/Users/alba.baldrich/Desktop/UNet/model_comparisions/LR_batch_evolution/predictions/' + (test_name) + '_binary_.png')
        
#         ix += 1

# # Read images and masks dataset and resize them
# # Read images and masks dataset
# for i in glob.glob(prova_image_path+'/*.jpg'):
#     test_image = cv2.imread(i)
#     test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
#     test_image = cv2.resize(test_image, dsize=(img_width, img_height),
#                         interpolation=cv2.INTER_NEAREST)
#     test_image = cv2.normalize(test_image, test_image, 0, 255, cv2.NORM_MINMAX)
#     prova_img_dataset.append(test_image)

# for i in glob.glob(prova_mask_path+'/*.png'):
#     test_mask = cv2.imread(i)
#     test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)
#     test_mask = cv2.resize(test_mask, dsize=(img_width, img_height),
#                       interpolation=cv2.INTER_NEAREST)
#     test_mask = cv2.normalize(test_mask, test_mask, 0, 255, cv2.NORM_MINMAX)
#     prova_msk_dataset.append(test_mask)

# # convert the list of arrays to a single numpy array
# prova_img_dataset = np.stack(prova_img_dataset)
# prova_msk_dataset = np.stack(prova_msk_dataset)

# X_test = prova_img_dataset
# y_test = prova_msk_dataset
# ########
# preds_test = model.predict(X_test, verbose=1)
# preds_test = (preds_test > threshold) * 255
# preds_test = preds_test.astype(np.uint8)
# ########
# # preds_test = model.predict(X_test, verbose=1)
# # preds_test_t = (preds_test > 0.5).astype(np.uint8)
# plot_sample(X_test, y_test, preds_test, preds_test_t)
'''
