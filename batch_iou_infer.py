import math
import numpy as np
from common.evadeUtil import calc_iou

'''
    批量iou推演过程
'''

# box1 三个人
# box2 四个闸机

box1 = [[0, 0, 1, 2],
        [0, 0, 1, 3],
        [1, 1, 2, 2]]
box2 = [[0, 0, 2, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 3],
        [1, 0, 3, 3]]

# 1.先判断每个人在哪个闸机下
print("1.先判断每个人在哪个闸机下")
iouList = calc_iou(box1, box2)

index = []
for iou in iouList:
    print(iou)
    index.append(iou.argmax())
print("index:", index)    # index值为闸机编号

# 2.再判断哪个闸机下有多个人
print("2.再判断哪个闸机下有多个人")
from collections import Counter
counter_result = Counter(index)
print(type(counter_result), counter_result)
resList = []
for res in counter_result.keys():
    if counter_result[res] > 1:
        resList.append(res)
        # print(res)    # 拿到几号闸机同时出现多人
print("几号闸机同时出现多人？：", resList)

# 3.从index中拿到哪些行是该通道的值
print("3.从index中拿到哪些行是该通道的值")
row_nums = []
for i in range(len(index)):
    if index[i] in resList:
        row_nums.append(i)
print("那些人出现在同一闸机下？：", row_nums)

unconfirm_evade = []
for i in range(len(box1)):
    if i in row_nums:
        unconfirm_evade.append(box1[i])
print(unconfirm_evade)

# 4.计算嫌疑人两两之间的距离
print("4.计算嫌疑人两两之间的距离")
center = [[left + (right - left) / 2,
           top + (bottom - top) / 2] for (left, top, right, bottom) in unconfirm_evade]
print(center)

for i in range(len(center)):
    for j in range(i + 1, len(center)):
        person1x, person1y = center[i][0], center[i][1]
        person2x, person2y = center[j][0], center[j][1]
        distance = math.sqrt(((person1x - person2x) ** 2) +
                             ((person1y - person2y) ** 2))

        print(center[i], center[j], distance)

        if distance > 1.0:    # 如果距离满足条件
            suspicion1 = unconfirm_evade[center.index(center[i])]    # 嫌疑人1
            suspicion2 = unconfirm_evade[center.index(center[j])]    # 嫌疑人2
            print(suspicion1, suspicion2, "涉嫌逃票")
            index1 = box1.index(suspicion1)
            index2 = box1.index(suspicion2)
            print("这两人真实序号：", index1, index2)
