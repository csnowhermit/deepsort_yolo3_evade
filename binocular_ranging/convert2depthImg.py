import numpy as np
import cv2
import binocular_ranging.camera_configs as camera_configs

'''
    转换成深度图
'''

cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 400, 0)
cv2.createTrackbar("num", "depth", 0, 10, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)

camera1 = cv2.VideoCapture(1)
# camera1.set(3, 320)
# camera1.set(4, 240)
camera2 = cv2.VideoCapture(2)
# camera2.set(3, 320)
# camera2.set(4, 240)



# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])    # [ 20202.172  16266.062 -61382.527]，输出是这个

cv2.setMouseCallback("depth", callbackFunc, None)

while True:
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()

    # print("frame1-frame2:", frame1.shape, frame2.shape)
    # cv2.resize(frame1, (480, 640))
    # cv2.resize(frame2, (480, 640))
    #
    # cv2.imshow("f1", frame1)
    # cv2.waitKey()
    # cv2.imshow("f2", frame2)
    # cv2.waitKey()

    if ret1 is False or ret2 is False:
        print(ret1, ret2)
        break

    # print(frame1.shape)
    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

    # print("img1_rectified-img2_rectified:", img1_rectified.shape, img2_rectified.shape)
    # cv2.imshow("ir1", img1_rectified)
    # cv2.waitKey()
    # cv2.imshow("ir2", img2_rectified)
    # cv2.waitKey()

    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    # 两个trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "depth")
    blockSize = cv2.getTrackbarPos("blockSize", "depth")
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5
    # blockSize = 3


    # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
    # numDisparities：最大视差值与最小视差值之差，窗口大小必须是16的整数倍
    # blockSize：匹配的块大小，必须在5-255之间，奇数
    stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
    # stereo = cv2.StereoBM_create(numDisparities=num, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., camera_configs.Q)
    # threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32), camera_configs.Q)
    # print("threeD:", threeD.shape, threeD)

    cv2.imshow("left", img1_rectified)
    cv2.imshow("right", img2_rectified)
    cv2.imshow("depth", disp)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("./snapshot/BM_left.jpg", imgL)
        cv2.imwrite("./snapshot/BM_right.jpg", imgR)
        cv2.imwrite("./snapshot/BM_depth.jpg", disp)

camera1.release()
camera2.release()
cv2.destroyAllWindows()