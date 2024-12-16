import torch
import os
import sys
# 将工作目录切换到项目根目录 自行填写项目路径
os.chdir('C:/CodeProjects/CarDetection')

# 添加 yolov5 到 Python 路径
# sys.path.append('./yolov5')
from yolov5 import train

def train_model():
    train.run(
        data='my_dataset/dataset_3/data.yaml',
        cfg='models/pre/yolov5s.yaml',
        weights='models/pre/yolov5s.pt',
        batch_size=16,
        epochs=100,
        img_size=640,
        project='runs/train',
        name='exp4',
        save_dir='models/exp4.pt'  # 保存模型的位置
    )

if __name__ == '__main__':
    train_model()
