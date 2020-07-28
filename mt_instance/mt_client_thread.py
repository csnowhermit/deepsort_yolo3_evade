import cv2
import time
import random
from common.config import log

'''
    视频接入线程
'''


def capture_thread(input_webcam, frame_buffer, ip, lock):
    if input_webcam == "0":
        input_webcam = int(0)
    print("capture_thread start: %s" % (input_webcam))
    log.logger.info("capture_thread start: %s" % (input_webcam))
    vid = cv2.VideoCapture(input_webcam)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    while True:
        try:
            return_value, frame = vid.read()
            time.sleep(random.uniform(0, 0.5))    # 随机延迟0-0.5s
        except Exception as e:
            time.sleep(0.5)    # 读取失败后直接重连没有任何意义
            vid = cv2.VideoCapture(input_webcam)
        if return_value is not True:
            break
        lock.acquire()
        frame_buffer.push((ip, frame))    # 缓冲区为tuple
        lock.release()
        cv2.waitKey(25)    # delay 25ms

