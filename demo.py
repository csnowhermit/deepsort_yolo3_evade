import os
import time
from collections import Counter
from common.config import normal_save_path, evade_save_path, ip
from common.dateUtil import formatTimestamp

which_gateList = [1, 2, 1, 2]

## 方法1：已废弃
# gate_status_list_new = []    # 闸机开关状态：只保留有人的通道
# gate_light_status_list_new = []    # 闸机灯状态：只保留有人通道的灯

# for i in range(len(passway_area_list)):
#     if i in which_gateList:
#         gate_status_list_new.append(gate_status_list[i])
#         gate_light_status_list_new.append(gate_light_status_list[i])

# 2.2、判断各自通道内的人数
gateCounter = Counter(which_gateList)
print("gateCounter:", gateCounter)

multi_personList = []    # 同时出现多人的闸机序列
for res in gateCounter.keys():
    if gateCounter[res] > 1:
        multi_personList.append(res)    # 拿到几号闸机同时出现多人

# pass_status_list = [0] * len(bboxes)    # 每个人的通行状态
pass_status_list = [0] * 4    # 每个人的通行状态

if len(multi_personList) > 0:    # 否则拿出各通道的人数，计算距离
    passwayPersonDict = {}    # <通道编号，通道内的人员list>
    for i in range(len(which_gateList)):    #
        if which_gateList[i] in multi_personList:
            if which_gateList[i] in passwayPersonDict.keys():
                tmp = passwayPersonDict.get(which_gateList[i])
                tmp.append(i)
                passwayPersonDict[which_gateList[i]] = tmp
            else:
                passwayPersonDict[which_gateList[i]] = [i]
    print("passwayPersonDict:", passwayPersonDict)

print(abs(-5))