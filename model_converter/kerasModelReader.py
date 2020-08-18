
from yolo import YOLO
import torch.nn as nn

if __name__ == '__main__':
    yolo = YOLO()
    print(yolo.yolo_model.to_json())    # 仅输出网络结构，不包含权值

    nn.Conv2d()
    nn.Conv1d()
    nn.Conv3d()