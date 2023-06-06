import os
import wandb

from utils import *
from dataset import CrearDataset, LoadTestDataset 
from train import ModelGeneration, TrainModel
from predict import SavePredictions, LoadModel
from evaluation import ShowPredictions, ShowTable, ConfusionMatrix

if __name__ == "__main__":
    EXPERIMENT = 'experiment_drop5'
        
    cfg = LoadParams('experiments/'+ EXPERIMENT + '.json')
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg["CUDA_VISIBLE_DEVICES"]
    os.environ["WANDB_API_KEY"] = '665e68bb4dd1224ff5144638d4a66a290f925520'
    wandb.login()

    TrainDataset, ValDataset = CrearDataset(cfg)

    run = wandb.init(project='smoke', name=cfg['test_name'], config=cfg)
    cfg = run.config
    
    model, callbacks = ModelGeneration(cfg, TrainDataset)
    model, history = TrainModel(model, callbacks, TrainDataset, ValDataset, cfg)
    
    # Load and preprocess the test dataset
    TestDataset = LoadTestDataset(cfg)
    ShowTable(model, TestDataset, "Conjunt de test")

    # Compute and save confusion matrix
    ConfusionMatrix(model, TestDataset, cfg, 'output/confusion_matrix/test/')
    ConfusionMatrix(model, ValDataset, cfg, 'output/confusion_matrix/validation/')
#    ShowPredictions(model, ValDataset)
    run.finish()



    # Save the predictions for the validation dataset
    SavePredictions(model, ValDataset, cfg, 'output/predictions/validation')
    
    # Save the predictions for the test dataset
    SavePredictions(model, TestDataset, cfg, 'output/predictions/test')