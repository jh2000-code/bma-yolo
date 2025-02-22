import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/BMA-YOLO.yaml')
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='1',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                #amp=False # close amp
                # fraction=0.2,
                project='',
                name='yolov8n',
                )