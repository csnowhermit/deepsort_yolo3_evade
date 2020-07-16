import cv2
import numpy as np

'''
    标定的结果
'''

# 左相机：内参矩阵，格式：
#   [[fx,0,u0]
# 	 [0,fy,v0]
# 	 [0,0,1]]
left_camera_matrix = np.array([[728.7339733, 0., 297.08895302],
                               [0., 706.88338989, 253.96508924],
                               [0., 0., 1.]])
left_distortion = np.array([[ 1.85746906e+00, -5.10658537e+01, 6.15935937e-03, 2.64645554e-02, 4.26522833e+02]])    # 畸变系数

# 右相机：内参矩阵
right_camera_matrix = np.array([[1.44478429e+03, 0.00000000e+00, 4.24573215e+02],
                               [0.00000000e+00, 1.39380099e+03, 2.37447406e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
right_distortion = np.array([[1.85746906e+00, -5.10658537e+01, 6.15935937e-03, 2.64645554e-02, 4.26522833e+02]])    # 畸变系数


om = np.array([0.01911, 0.03125, -0.00960]) # 旋转关系向量（该变量只有计算R才用到）
# R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
# print("R:", R)
R = np.array([[0.9994657, 0.00989626, 0.03115082],
              [-0.00929915, 0.99977135, -0.01925542],
              [-0.03133425, 0.01895545, 0.9993292]])    # 旋转矩阵

# R = np.array([0.9994657, 0.00989626, 0.03115082])
T = np.array([-10.58179367, 4.46541288, 21.38186075]) # 平移关系向量
# T = np.array([[-10.58179367],
#               [4.46541288],
#               [21.38186075]])

size = (640, 480) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion,
                                                                  size, R, T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)