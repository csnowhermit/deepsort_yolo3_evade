import os
import cv2
import time
from common.dateUtil import formatTimestamp

'''
    海康摄像机：子码流接入
'''

cap = cv2.VideoCapture("rtsp://admin:quickhigh123456@192.168.120.155/h264/ch1/sub/av_stream")    # 子码流
savepath = "D:/workspace/testData/"

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    print("frame:", type(frame), frame.shape)    # <class 'numpy.ndarray'> (480, 640, 3)

    cv2.imshow("", frame)
    cv2.waitKey(1)

    cv2.imwrite(os.path.join(savepath, formatTimestamp(time.time()) + ".jpg"), frame)
