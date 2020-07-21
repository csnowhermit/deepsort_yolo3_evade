import re
import json
import traceback
from common.config import *

'''
    数据库操作工具
'''

'''
    批量入库：识别明细数据
    :param ip 哪个摄像头
    :param curr_time 当前时间：%Y-%m-%d_%H:%M:%S
    :param save_file 图片保存路径
    :param read_time 读取耗时，s
    :param detect_time 检测耗时，s
    :param predicted_class 检测类别：head/person
    :param TrackContentList 被追踪人的明细
    :return 
'''
def saveManyDetails2DB(ip, curr_time, savefile, read_time, detect_time, predicted_class, TrackContentList):
    if table_exists(table_name) is False:
        create_detail_info_table(table_name)

    for trackContent in TrackContentList:
        try:
            sql = "insert into %s" % (table_name)
            sql = sql + '''
                            (curr_time, savefile, pass_status, read_time, detect_time, 
                            predicted_class, score, box, person_id, trackState, 
                            ip, gate_num, direction) 
                            VALUES ('%s', '%s', '%s', %f, %f, 
                                    '%s', %f, '%s', %d, %d, '%s', '%s', '%s')
                        ''' % (curr_time, savefile, trackContent.pass_status, read_time, detect_time,
                               predicted_class, trackContent.score, trackContent.bbox, trackContent.track_id,
                               trackContent.state,
                               ip, trackContent.gate_num, trackContent.direction)
            cursor.execute(sql)
            conn.commit()
            log.logger.info("%s %s %s " % (curr_time, savefile, json.dumps(obj=trackContent.__dict__, ensure_ascii=False)))
        except Exception as e:
            log.logger.error(traceback.format_exc())
            log.logger.error(sql)
            conn.rollback()


'''
    判断表是否存在
    :param table_name 表名
    :return True，表存在；False，表不存在
'''
def table_exists(table_name):
    sql = "show tables;"
    cursor.execute(sql)
    tables = [cursor.fetchall()]
    table_list = re.findall('(\'.*?\')',str(tables))
    table_list = [re.sub("'",'',each) for each in table_list]
    if table_name in table_list:
        return True
    else:
        return False

'''
    创建信息明细表
    :param ip 用于组装表名：details_ip
    :return ret 0，创建成功
'''
def create_detail_info_table(table_name):
    sql = '''
        CREATE TABLE `%s`  (
            `curr_time` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '当前时刻，精确到s',
            `savefile` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '保存文件路径',
            `pass_status` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '通过状态：0正常通过，1涉嫌逃票',
            `read_time` float(10, 5) NULL DEFAULT NULL COMMENT '读取耗时',
            `detect_time` float(10, 5) NULL DEFAULT NULL COMMENT '检测耗时',
            `predicted_class` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '检测类别',
            `score` float(10, 5) NULL DEFAULT NULL COMMENT '得分值',
            `box` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '人头框，左上右下',
            `person_id` int(10) NULL DEFAULT NULL COMMENT '人物id',
            `trackState` int(2) NULL DEFAULT NULL COMMENT '确认状态：1未确认，2已确认',
            `ip` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '摄像机ip',
            `gate_num` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '闸机编号',
            `direction` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '方向：0出站，1进站'
        ) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;
    ''' % (table_name)

    ret = cursor.execute(sql)
    log.logger.info("%s 表已创建: %s" % (table_name, ret))
    return ret

if __name__ == '__main__':
    print(table_exists("details_10.6.8.181"))
    print(table_exists("details_10.6.8.222"))

    if table_exists(table_name) is False:
        print(create_detail_info_table(table_name))
    else:
        print(table_name + " 表已存在")