import numpy as np
import cv2    
import matplotlib.pyplot as plt
import torch
import glob
import os

from utils import *
from frames import ExtractFrames

EXPERIMENT = 'video_segmentation'
        
cfg = LoadParams('experiments/'+ EXPERIMENT + '.json')
os.environ["CUDA_VISIBLE_DEVICES"]=cfg["CUDA_VISIBLE_DEVICES"]


#Load video path
video_path = os.path.join(cfg['video_dataset'], cfg['video_name']  + cfg['form_video'])

