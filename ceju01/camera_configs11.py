import cv2
import numpy as np

'''
    标定的结果
'''

# 左相机：内参矩阵，格式：
#   [[fx,0,u0]
# 	 [0,fy,v0]
# 	 [0,0,1]]
left_camera_matrix = np.array([[517.73610256, 0., 359.68192807],
                               [0., 503.18730169, 227.19967664],
                               [0., 0., 1.]])
left_distortion = np.array([[-0.29337933, -0.41057414, -0.0052619, -0.01017261, 1.01789224]])    # 畸变系数

# 右相机：内参矩阵
right_camera_matrix = np.array([[572.08224964, 0., 345.11181113],
                                [0., 545.096531, 188.08904018],
                                [0., 0., 1.]])
right_distortion = np.array([[-0.41458153, 0.09833357, 0.01798432, -0.00493075, 0.05291643]])    # 畸变系数


# om = np.array([0.01911, 0.03125, -0.00960]) # 旋转关系向量（该变量只有计算R才用到）
# R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
# print("R:", R)
R = np.array([[0.99885449, -0.00103767, -0.04783961],
              [0.00305682, 0.9991065, 0.0421528 ],
              [0.04775312, -0.04225075, 0.99796519]])    # 旋转矩阵：3*3

# R = np.array([0.9994657, 0.00989626, 0.03115082])
# T = np.array([-1.15831903, 0.06849329, -0.80142875]) # 平移关系向量：3*1
T = np.array([[-1.1257236],
              [0.05906977],
              [-0.88203856]])

# size = (640, 480) # 图像尺寸，应写成（h, w）
size = (480, 640)

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion,
                                                                  size, R, T)
# print("Q:", Q)
# print("Q[2,3]:", Q[2,3])
# 焦距：一般取立体校正后的重投影矩阵Q中的 Q[2,3]
focal_length = 524.141916345

# 基线距离：一般取平移矩阵的第一个值的绝对值
baseline = 1.1257236


# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)