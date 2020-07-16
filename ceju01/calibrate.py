import os
import time
import cv2
import numpy as np

'''
    对getImage.py拍摄的照片进行标定
'''

# 0.基本配置
show_corners = False    # 展示角点

image_number = 13
board_size = (7, 5)  # 8*6的棋盘
square_Size = 20

image_lists = []  # 存储获取到的图像
image_points = []  # 存储图像的点

# 1.读图,找角点
image_dir = "./snapshot/"
image_names = []

[image_names.append(image_dir + "/left_%d.jpg" % i) for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
[image_names.append(image_dir + "/right_%d.jpg" % i) for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
print(len(image_names))

for image_name in image_names:
    print(image_name)
    image = cv2.imread(image_name, 0)
    found, corners = cv2.findChessboardCorners(image, board_size)  # 粗查找角点
    if not found:
        print("ERROR(no corners):" + image_name)
        continue

    # 展示结果
    if show_corners:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, board_size, corners, found)
        cv2.imwrite(image_name.split(os.sep)[-1], vis)
        cv2.namedWindow("xxx", cv2.WINDOW_NORMAL)
        cv2.imshow("xxx", vis)    # 展示标定的角点
        cv2.waitKey()
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), term)  # 精定位角点
    image_points.append(corners.reshape(-1, 2))
    image_lists.append(image)

# 2. 构建标定板的点坐标，objectPoints
object_points = np.zeros((np.prod(board_size), 3), np.float32)
object_points[:, :2] = np.indices(board_size).T.reshape(-1, 2)
object_points *= square_Size
object_points = [object_points] * image_number

# object_points = np.repeat(object_points[np.newaxis, :], 13, axis=0)
# print(object_points.shape)

# 3. 分别得到两个相机的初始CameraMatrix
h, w = image_lists[0].shape
camera_matrix = list()

camera_matrix.append(cv2.initCameraMatrix2D(object_points, image_points[:image_number], (w, h), 0))
camera_matrix.append(cv2.initCameraMatrix2D(object_points, image_points[image_number:], (w, h), 0))

# 4. 双目视觉进行标定
term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(object_points, image_points[:image_number], image_points[image_number:], camera_matrix[0],
                        None, camera_matrix[1], None, (w, h),
                        flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS |
                              cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5,
                        criteria=term)

print("左相机内参矩阵：cameraMatrix1：", cameraMatrix1)
print("左相机畸变系数：distCoeffsl：", distCoeffs1)
print("右相机内参矩阵：cameraMatrix2：", cameraMatrix2)
print("右相机畸变系数：distCoeffs2：", distCoeffs2)
print("旋转关系矩阵R：", R)
print("平移关系矩阵T：", T)

# 5. 标定精度的衡量， TODO

# 6. 保存标定结果 TODO

# 7. 矫正一张图像看看，是否完成了极线矫正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
    cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w, h), R, T)

map1_1, map1_2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_16SC2)
map2_1, map2_2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_16SC2)

start_time = time.time()
result1 = cv2.remap(image_lists[0], map1_1, map1_2, cv2.INTER_LINEAR)
result2 = cv2.remap(image_lists[image_number], map2_1, map2_2, cv2.INTER_LINEAR)
print("变形处理时间%f(s)" % (time.time() - start_time))

result = np.concatenate((result1, result2), axis=1)
result[::20, :] = 0
cv2.imwrite("rec.png", result)