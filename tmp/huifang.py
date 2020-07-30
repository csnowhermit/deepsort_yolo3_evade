import os
import cv2
import time
import hashlib
import threading
from common.dateUtil import formatTimestamp

# url = "rtsp://admin:quickhigh123456@192.168.120.155/Streaming/tracks/101?starttime=20200730T134015Z&endtime=20200730T134515Z"

url = "rtsp://admin:quickhigh123456@192.168.120.155/h264/ch1/sub/av_stream"

def cap_thread(input_webcam, frame_buffer, img_cached, lock):
    if input_webcam == "0":
        input_webcam = int(0)
    print("capture_thread start: %s" % (input_webcam))

    vid = cv2.VideoCapture(input_webcam)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    while True:
        return_value, frame = vid.read()

        lock.acquire()
        frame_buffer.append(frame)
        img_cached.append(frame)

        print("cap_thread:", len(frame_buffer), len(img_cached))

        if len(frame_buffer) > 30*60*10:
            frame_buffer.remove(frame_buffer[0])
        if len(img_cached) > 30*60*10:
            img_cached.remove(img_cached[0])

        lock.release()
        cv2.waitKey(25)    # delay 25ms

def consumer_thread(frame_buffer, img_cached, lock):
    num = 0
    while True:
        if len(frame_buffer) > 0:
            lock.acquire()
            # frame = frame_buffer.pop()  # 每次拿最新的
            frame = frame_buffer[len(frame_buffer) - 1]
            lock.release()

            num += 1
            cc = 200

            if num < cc:
                continue

            if len(img_cached) > cc * 2:    # 这里是要截的视频，前后各截取100帧
                tmp = img_cached[cc-100: cc+100]

                video_FourCC = 875967080
                video_fps = 25
                video_size = (640, 480)

                output_path = os.path.join("D:/monitor_images/10.6.8.181/evade_video/20200730/", "10.6.8.181_" + formatTimestamp(time.time(), format='%Y%m%d_%H%M%S', ms=True) + ".mp4")
                print(output_path, len(tmp), len(img_cached))

                out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
                for t in tmp:
                    print("t:", t.shape)
                    out.write(t)
                print("保存阶段视频完成")
                exit(1)
        else:
            # print("++++++++++++++++++")
            pass





if __name__ == '__main__':
    frame_buffer = []
    img_cached = []
    lock = threading.RLock()
    t1 = threading.Thread(target=cap_thread, args=(url, frame_buffer, img_cached, lock))
    t1.start()
    time.sleep(10)
    t2 = threading.Thread(target=consumer_thread, args=(frame_buffer, img_cached, lock))
    t2.start()