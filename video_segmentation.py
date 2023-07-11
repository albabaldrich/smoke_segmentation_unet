from predict import LoadModel
from utils import *
from video_prediction import *

video_params = 'video_segmentation'
EXPERIMENT = 'final_unet_model'
cfg = LoadParams('experiments/'+ EXPERIMENT + '.json')
cfg1 = LoadParams('experiments/'+ video_params + '.json')
os.environ["CUDA_VISIBLE_DEVICES"]=cfg["CUDA_VISIBLE_DEVICES"]

cfg = {**cfg1, **cfg}
model = LoadModel(cfg)
VideoPred(model, cfg)

