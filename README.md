# smoke_segmentation_unet
The folder structure for this U-Net algorithm application is the following:
–	smoke_dataset
    o	train (175 images - afterward divided into 90% train and 10% test)
        ■	images
        ■	masks
    o	test (15 images)
        ■	images
        ■	masks
–	video_dataset
–	experiments
    o	experiment_name.json
    o	video_segmentation.json
–	code
    o	main.py
    o	utils.py
    o	dataset.py
    o	unet_model.py
    o	train.py
    o	predict.py
    o	evaluation.py
    o	video_prediction.py
    o	video_segmentation.py
–	output
    o	model
    o	confusion matrix
    o	prediction
    o	video_output
