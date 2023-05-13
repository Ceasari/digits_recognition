from ultralytics import YOLO

# Train parameters
params = {
    'data': 'datasets/new_data/new_data.yaml',  # path to YAML file with datasets information
    'epochs': 5,  # number of epochs to train for
    'patience': 0,  # epochs to wait for no observable improvement for early stopping of training
    'batch': 16,  # number of images per batch (-1 for AutoBatch)
    'save': True,  # save train checkpoints and predict results
    'save_period': -1,  # Save checkpoint every x epochs (disabled if < 1)
    'workers': 8,  # number of worker threads for data loading (per RANK if DDP)
    'optimizer': 'SGD',  # optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
    'lr0': 0.01,  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    'lrf': 0.01,  # final learning rate (lr0 * lrf)
    'momentum': 0.937,  # SGD momentum/Adam beta1
    'weight_decay': 0.0005,  # optimizer weight decay 5e-4
    'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
    'warmup_momentum': 0.8,  # warmup initial momentum
    'warmup_bias_lr': 0.1,  # warmup initial bias lr
    'val': True,  # validate/test during training
}

# First train, will download pretrained weights from the model you specified
# Will create folder RUN with the first start time to save train results and weights
model = YOLO('yolov8n.pt')  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt - will be downloaded


# Resume training from 'runs/train/exp/weights/ specified 'weights/best.pt' or 'weights/last.pt'
model = YOLO('runs/detect/train/weights/best.pt')  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt


results = model.train(**params)