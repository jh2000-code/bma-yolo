import warnings
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    model.val(data=r'',
              split='test',
              batch=1,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='',
              name='yolov8n',
              )