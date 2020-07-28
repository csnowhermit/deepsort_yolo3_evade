import threading
from common.Stack import Stack
from mt_instance.mt_client_thread import capture_thread
from mt_instance.mt_detect_thread import detect_thread


if __name__ == '__main__':
    frame_buffer = Stack(30 * 10)
    lock = threading.RLock()

    webcam1 = "E:/BaiduNetdiskDownload/2020-04-14/10.6.8.181_01_20200414185435204.mp4"
    webcam2 = "E:/BaiduNetdiskDownload/2020-04-14/10.6.8.222_01_20200414191610180.mp4"

    t1 = threading.Thread(target=capture_thread, args=(webcam1, frame_buffer, "10.6.8.181", lock))
    t1.start()
    t2 = threading.Thread(target=capture_thread, args=(webcam2, frame_buffer, "10.6.8.222", lock))
    t2.start()

    t3 = threading.Thread(target=detect_thread, args=(frame_buffer, lock))
    t3.start()