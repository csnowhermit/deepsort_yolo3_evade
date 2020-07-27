import os
import pymysql
from common.Logger import Logger as Logger

'''
    本实例的配置项
'''

# 注册中心
zkurl = "127.0.0.1:2181"

# 连接摄像头
ip = "10.6.8.181"
rtsp_url = "rtsp://admin:quickhigh123456@192.168.120.155/h264/ch1/sub/av_stream"    # 用子码流读取

# 图像大小
# image_size = "1920x1080"
image_size = (1920, 1080)    # 图片大小

# 数据库
conn = pymysql.connect(host='127.0.0.1',
                       port=3306,
                       user='root',
                       password='123456',
                       database='evade',
                       charset='utf8mb4')
cursor = conn.cursor()

table_name = "details_%s" % (ip.replace(".", "_"))    # 表名：正常+逃票
evade_table_name = "evade_details"    # 逃票表（所有摄像头都存一张表）

# 需特殊处理的类别
special_types = ['head', 'person']
tracker_type = 'head'    # 需要tracker的类别

# 保存路径
normal_save_path = "D:/monitor_images/" + ip + "/normal_images/"
evade_save_path = "D:/monitor_images/" + ip + "/evade_images/"
evade_origin_save_path= "D:/monitor_images/" + ip + "/evade_origin_images/"    # 保存检出逃票的原图

if os.path.exists(normal_save_path) is False:
    os.makedirs(normal_save_path)
if os.path.exists(evade_save_path) is False:
    os.makedirs(evade_save_path)
if os.path.exists(evade_origin_save_path) is False:
    os.makedirs(evade_origin_save_path)

# 通过状态的判断条件：图像的高*比例，在两比例之间，认定为涉嫌逃票
up_distance_rate = 0.6
down_distance_rate = 0.2


# 日志文件
logfile = 'D:/evade_logs/evade_%s.log' % ip
log = Logger(logfile, level='info')

# 检出框与tracker框的iou，解决人走了框还在的情况
track_iou = 0.45