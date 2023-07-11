# smoke_segmentation_unet
The folder structure for this U-Net algorithm application is the following:
- smoke_dataset
    - train (175 images - afterward divided into 90% train and 10% test)
        - images
        - masks
    - test (15 images)
        - images
        - masks

- video_dataset

- experiments
    - experiment_name.json
    - video_segmentation.json

- code
    - main.py
    - prediction_output.py
    - utils.py
    - dataset.py
    - unet_model.py
    - unet_model_original.py
    - train.py
    - hyper_search.py
    - predict.py
    - evaluation.py
    - video_prediction.py
    - video_segmentation.py

- output
    - model
    - confusion matrix
    - prediction
    - video_output
