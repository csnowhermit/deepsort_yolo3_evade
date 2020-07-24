import cv2

'''
    双目摄像头，无畸变
'''

cap = cv2.VideoCapture(2)
print(cap.isOpened())
while True:
    ret1, frame1 = cap.read()
    cv2.imshow("1", frame1)
    cv2.waitKey(1)