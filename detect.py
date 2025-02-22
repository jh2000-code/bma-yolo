import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/weights/best.pt') # select your model.pt path
    model.predict(source=r'',
                project=r'',
                name=r'',
                save=True,
                # visualize=True # visualize model features maps
                )