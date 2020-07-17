import os
import pymysql

'''
    本实例的配置项
'''

# 注册中心
zkurl = "127.0.0.1:2181"

# 连接摄像头
ip = "10.6.8.181"
rtsp_url = "rtsp://admin:quickhigh123456@192.168.120.155/h264/ch1/sub/av_stream"    # 用子码流读取

# 数据库
conn = pymysql.connect(host='127.0.0.1',
                       port=3306,
                       user='root',
                       password='123456',
                       database='evade',
                       charset='utf8mb4')
cursor = conn.cursor()

# 需特殊处理的类别
special_types = ['head', 'person']
tracker_type = 'head'    # 需要tracker的类别

# 保存路径
normal_save_path = "D:/monitor_images/" + ip + "/normal_images/"
evade_save_path = "D:/monitor_images/" + ip + "/evade_images/"

if os.path.exists(normal_save_path) is False:
    os.makedirs(normal_save_path)
if os.path.exists(evade_save_path) is False:
    os.makedirs(evade_save_path)

# 通过状态的判断条件：图像的高*比例，在两比例之间，认定为涉嫌逃票
up_distance_rate = 0.6
down_distance_rate = 0.18


