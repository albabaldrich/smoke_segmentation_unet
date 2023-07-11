import glob
import cv2
import random
import matplotlib.pyplot as plt
import albumentations as A
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle

##############################################################################
def LoadForPrediction(cfg, path, ext='*.jpg'):
    def resize(im):
        return cv2.resize(im, dsize=(cfg['img_width'], cfg['img_height']), interpolation=cv2.INTER_NEAREST)
    
    image_dataset = []
    # Read images and masks dataset and normalize them
    for name in glob.glob( os.path.join(path,ext) ):
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        image_dataset.append(resize(image))
    return {'images': np.array(image_dataset).astype(np.float32)/255.0, 'names':glob.glob( os.path.join(path,ext) )}

def LoadImages (cfg, mode='train'):
    if mode == 'train':
        path = 'train_path'
    else:
        path = 'test_path'
        
    image_dataset = []
    mask_dataset = []
    # Read images and masks dataset and normalize them
    for name in glob.glob( os.path.join(cfg[path],'images','*.jpg') ):
        basename = os.path.splitext(os.path.basename(name))[0]
        image = cv2.imread(os.path.join(cfg[path],'images',basename+'.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, dsize=(img_width, img_height),
        #                   interpolation=cv2.INTER_NEAREST)
        image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        image_dataset.append(image)

        mask = cv2.imread(os.path.join(cfg[path],'masks',basename+'.png'))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #mask = cv2.resize(mask, dsize=(img_width, img_height),
        #                  interpolation=cv2.INTER_NEAREST)
        mask = cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX)
        mask_dataset.append(np.expand_dims(mask, axis=-1))
        
    return {'images':image_dataset, 'masks':mask_dataset}


def ResizeDataset(Dataset, cfg):
    def resize(im):
        return cv2.resize(im, dsize=(cfg['img_width'], cfg['img_height']), interpolation=cv2.INTER_NEAREST)
    imgs = []
    msks = []
    for image,mask in zip(Dataset['images'], Dataset['masks']):
        imgs.append(resize(image))
        msks.append(resize(mask))
    return {'images':imgs, 'masks':msks}

def DatasetToFloat(Dataset):
    return {'images': np.array(Dataset['images']).astype(np.float32)/255.0, 'masks': np.array(Dataset['masks']).astype(np.float32)/255.0}


def DataAugmentation(Dataset, cfg):
    augm_img = []
    augm_msk = []

    train_augment = A.Compose([A.RandomResizedCrop (width=cfg['img_width'], height=cfg['img_height'], scale=[0.07, 1.0], ratio=(1, 1), p=0.8),
                                A.GaussNoise(p=0.2),
                                A.Blur(p=0.1),
                                A.HorizontalFlip(p=0.5),
                                A.RandomBrightnessContrast(p=0.2),
                                A.RandomShadow(p=0.1)
                                ],p=1)

    for image,mask in zip(Dataset['images'], Dataset['masks']):
        done = False
        while (not done):
            try: 
                #train_augm_data = train_augment(image=image1, mask=mask1)
                train_augm_data = train_augment(image=image, mask=mask)
                done = True
            except:
                pass

        augm_img.append(train_augm_data['image'])
        augm_msk.append(train_augm_data['mask'])
        
    return {'images':augm_img, 'masks':augm_msk}

    

def CrearDataset(cfg):
    Dataset = LoadImages(cfg)

    train_images, val_images, train_masks, val_masks = train_test_split(Dataset['images'], Dataset['masks'], test_size=cfg['test_size'], shuffle=False)
    train_images, train_masks = shuffle(train_images, train_masks)
    TrainDataset = {'images':train_images, 'masks':train_masks}
    ValDataset  = {'images':val_images,   'masks':val_masks}

    AugmentedDataset = TrainDataset
    for i in range(cfg['num_augmentations']):
        AugDat = DataAugmentation(TrainDataset, cfg)
        AugmentedDataset = {'images': AugmentedDataset['images']+AugDat['images'], 'masks': AugmentedDataset['masks']+AugDat['masks']}
    
    AugmentedDataset = ResizeDataset(AugmentedDataset, cfg)
    ValDataset = ResizeDataset(ValDataset, cfg)

    # Check if augmented images and masks are the corresponding one with each other
    #for i in range (10):
    #    ShowDatasetExample(AugmentedDataset, TrainDataset, i)
 

    AugmentedDataset = DatasetToFloat(AugmentedDataset)
    ValDataset = DatasetToFloat(ValDataset)
    
    return AugmentedDataset, ValDataset


def LoadTestDataset (cfg):
    TestDataset = LoadImages(cfg, 'test')
    TestDataset = ResizeDataset(TestDataset, cfg)
    TestDataset = DatasetToFloat(TestDataset)
        
    return TestDataset



def ShowDatasetExample(Augmented, Dataset=None, i=-1):
    if i==-1:
        image_number = random.randint(0, len(Augmented['images']))
    else:
        image_number = i
    f = plt.figure()
    if Dataset:
        plt.subplot(221)
        plt.imshow((Augmented['images'][image_number]), cmap='gray')
        plt.subplot(222)
        plt.imshow((Augmented['masks'][image_number]), cmap='gray')
        plt.subplot(223)
        plt.imshow((Dataset['images'][0]), cmap='gray')
        plt.subplot(224)
        plt.imshow((Dataset['masks'][0]), cmap='gray')
    else:
        plt.subplot(121)
        plt.imshow((Augmented['images'][image_number]), cmap='gray')
        plt.subplot(122)
        plt.imshow((Augmented['masks'][image_number]), cmap='gray')
        
    for ax in f.axes:
        ax.axison = False    
    plt.savefig('figures/borrar'+str(i)+'.png')
    #plt.show()
    plt.close('all')
