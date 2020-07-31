import cv2

# url = "rtsp://admin:quickhigh123456@192.168.120.155/Streaming/tracks/101?starttime=19700103T020150Z&endtime=19700103T020200Z"
#
# rtsp_url = "rtsp://admin:quickhigh123456@192.168.120.155/h264/ch1/sub/av_stream"    # 用子码流读取
#
# cap = cv2.VideoCapture(url)
# print(cap.isOpened())
# while True:
#       ret, frame = cap.read()
#       cv2.imshow("", frame)
#       cv2.waitKey(1)


list = ['a', 'b', 'c', 'd', 'e']
print(list.index('f'))
print(list.__contains__('f'))