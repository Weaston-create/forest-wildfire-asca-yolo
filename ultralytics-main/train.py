import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\yolo\0224\ultralytics-main\datasets\yolo26-Gaihou.yaml')
    model.train(data=r'D:\yolo\0224\ultralytics-main\datasets\data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                patience=0,
                optimizer='MuSGD',
                amp=False,
                project='runs/train',
                name='YOLO26-Gaihou',
                )






