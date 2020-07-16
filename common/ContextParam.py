import os
from common.config import cursor
from common.entity import CapLocation

'''
    获取上下文参数
    1.该ip下通道、闸机门、等有效对应位置
'''

'''
    获取指定摄像头的ContextParam
'''
def getContextParam():
    sql = '''
            select gate_num, direction, default_direct, entrance,
                   entrance_direct, entrance_gate_num, displacement,
                   passway_area, gate_area, gate_light_area 
            from cap_location where is_enabled='y'
    '''
    cursor.execute(sql)

    results = cursor.fetchall()

    capLocationList = []

    for row in results:
        print(row)
        print(row[0])
        capLocationList.append(CapLocation(row[0], row[1], row[2], row[3], row[4],
                                           row[5], row[6], row[7], row[8], row[9]))
    return capLocationList


