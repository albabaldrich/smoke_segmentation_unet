import os
import wandb

from utils import *
from dataset import CrearDataset, LoadTestDataset 
from train import ModelGeneration, TrainModel
from predict import SavePredictions, LoadModel
from evaluation import ShowPredictions, ShowTable, ConfusionMatrix


def GetHyperparameters(cfg):
    if not any([isinstance(cfg[k], dict) and 'sweep' in cfg[k].keys() for k in cfg.keys()]):
        return cfg, None
    sweep = {}
    remove=[]
    for k in cfg.keys():
        if isinstance(cfg[k], dict) and 'sweep' in cfg[k].keys():
            sweep[k]=cfg[k]['sweep']
            remove.append(k)
    for k in remove:
        cfg.pop(k)
    return cfg, sweep

def HyperparameterSearch(cfg, name):
    # Define sweep config
    sweep_configuration = {    
        'method': 'bayes',
        'name': name +'_sweep',
        'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
        'parameters': cfg
    }
                
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='smoke')
    return sweep_id

def main():
    global TrainDataset, ValDataset, sweep, cfg
    if sweep:
        run = wandb.init()
    for k in wandb.config.keys():
        cfg[k] = wandb.config[k]
    model, callbacks = ModelGeneration(cfg, TrainDataset)
    model, history = TrainModel(model, callbacks, TrainDataset, ValDataset, cfg)
    return model
    

if __name__ == "__main__":
    global TrainDataset, ValDataset, sweep
    EXPERIMENT = 'sweep_4'
    cfg = LoadParams('experiments/'+ EXPERIMENT + '.json')
    cfg, sweep = GetHyperparameters(cfg)

    os.environ["CUDA_VISIBLE_DEVICES"]=cfg["CUDA_VISIBLE_DEVICES"]
    os.environ["WANDB_API_KEY"] = 'WRITE YOUR WANDB KEY'

    TrainDataset, ValDataset = CrearDataset(cfg)
    
    wandb.login()
    if not sweep:
        run = wandb.init(project='smoke', name=cfg['test_name'], config=cfg)
        model = main()    
    else:
        sweep_id = HyperparameterSearch(sweep, cfg['test_name'])
        wandb.agent(sweep_id, function=main, count = 50)
    