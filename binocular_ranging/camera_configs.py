import cv2
import numpy as np

'''
    标定的结果
'''

# 左相机：内参矩阵，格式：
#   [[fx,0,u0]
# 	 [0,fy,v0]
# 	 [0,0,1]]
left_camera_matrix = np.array([[447.90380859,   0. ,        348.33380727],
 [  0.    ,     430.53915405, 242.06905612],
 [  0.   ,        0.       ,    1.        ]])
left_distortion = np.array([[-0.36507462, -0.27302447, -0.00897782, -0.01808996 , 1.03724571]])    # 畸变系数

# 右相机：内参矩阵
right_camera_matrix = np.array([[427.47573853  , 0.   ,      357.7419588 ],
 [  0.      ,   417.95352173, 238.19509014],
 [  0.       ,    0.      ,     1.        ]])
right_distortion = np.array([[-0.33707202, -0.12083875, -0.00560685, -0.00911914 , 0.40913414]])    # 畸变系数


# om = np.array([0.01911, 0.03125, -0.00960]) # 旋转关系向量（该变量只有计算R才用到）
# R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
# print("R:", R)
R = np.array([[ 0.99983132 , 0.0104384 , -0.01511183],
 [-0.01017167,  0.999793 ,   0.01762082],
 [ 0.01529263, -0.01746413 , 0.99973053]])    # 旋转矩阵：3*3

# R = np.array([0.9994657, 0.00989626, 0.03115082])
# T = np.array([-1.1257236, 0.05906977, -0.88203856]) # 平移关系向量：3*1
T = np.array([[-1.52917121],
 [-0.12382173],
 [-0.6491331 ]])

size = (640, 480) # 图像尺寸，应写成（h, w）
# size = (480, 640)

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