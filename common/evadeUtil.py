import os
import math
import numpy as np
from collections import Counter
from common.config import up_distance_rate, down_distance_rate
from common.ContextParam import getContextParam
from common.entity import TrackContent

'''
    逃票检测工具类
'''

# cap与location的绑定关系
capLocationList = getContextParam()
passway_area_list = [[capLocation.passway_area.left,
                      capLocation.passway_area.top,
                      capLocation.passway_area.right,
                      capLocation.passway_area.bottom] for capLocation in capLocationList]
passway_default_direct_list = [capLocation.default_direct for capLocation in capLocationList]    # 闸机的方向：0出站，1进站
print("passway_default_direct_list:", passway_default_direct_list)    # 实时计算拿默认方向，批处理用人相对位置结合cap_location.displacement字段微调


'''
    拿到指定通道的默认方向
    :param gate_num int类型，闸机编号
    :return 返回该通道的默认方向：0出站，1进站
'''
def getDefaultDirection(gate_num):
    for capLocation in capLocationList:
        if capLocation.gate_num == str(gate_num):
            return capLocation.default_direct

'''
    逃票判定
    :param tracks 人的track
    :param other_classes 其他的类别
    :param other_boxs 其他类别框，上左下右
    :param other_scores 其他类别的得分值
    :param height 图像的高
    :return flag, TrackContentList 通行状态，新的追踪人的内容
'''
def evade_vote(tracks, other_classes, other_boxs, other_scores, height):
    TrackContentList = []    # 追踪人的内容，新增闸机编号和通过状态
    flag = "NORMAL"    # 默认该帧图片的通行状态为NORMAL，遇到有逃票时改为WARNING
    up_distance_threshold = height * up_distance_rate
    down_distance_threshold = height * down_distance_rate

    # bboxes = [track.to_tlbr() for track in tracks]    # 所有人的人物框
    bboxes = [[int(track.to_tlbr()[0]),
               int(track.to_tlbr()[1]),
               int(track.to_tlbr()[2]),
               int(track.to_tlbr()[3])] for track in tracks]  # 所有人的人物框

    if len(bboxes) < 1:
        flag = "NO_ONE"
        return flag, TrackContentList    # 没人的话返回标识“NO_ONE”

    # 1.判断各自在哪个通道内
    which_gateList = isin_which_gate(bboxes)

    # 2.判断各自通道内的人数
    gateCounter = Counter(which_gateList)

    multi_personList = []
    for res in gateCounter.keys():
        if gateCounter[res] > 1:
            multi_personList.append(res)    # 拿到几号闸机同时出现多人

    pass_status_list = []    # 每个人的通行状态

    if multi_personList is None or len(multi_personList) < 1:    # 如果没有出现同一通道出现多人的情况，则认为都是 0正常通行
        pass_status_list = [0 for i in range(len(bboxes))]
    else:    # 否则拿出各通道的人数，计算距离
        row_nums = []
        for i in range(len(which_gateList)):
            if which_gateList[i] in multi_personList:
                row_nums.append(i)
        print("那些人出现在同一闸机下？：", row_nums)

        suspicion_evade = []    # 逃票嫌疑list
        for i in range(len(bboxes)):
            if i in row_nums:
                suspicion_evade.append(bboxes[i])

        # 3.计算两两之间的距离，通过距离判断是否属于逃票
        center = [[left + (right - left) / 2,
                   top + (bottom - top) / 2] for (left, top, right, bottom) in suspicion_evade]
        evade_index_list = []    # 涉嫌逃票的序号：在原始bboxes中的序号
        for i in range(len(center)):
            for j in range(i + 1, len(center)):
                person1x, person1y = center[i][0], center[i][1]
                person2x, person2y = center[j][0], center[j][1]
                distance = math.sqrt(((person1x - person2x) ** 2) +
                                     ((person1y - person2y) ** 2))

                print(center[i], center[j], distance)

                if distance >= down_distance_threshold and distance <= up_distance_threshold:  # 如果距离满足条件
                    suspicion1 = suspicion_evade[center.index(center[i])]  # 嫌疑人1
                    suspicion2 = suspicion_evade[center.index(center[j])]  # 嫌疑人2
                    # print(suspicion1, suspicion2, "涉嫌逃票")    # [0, 0, 1, 2] [1, 1, 2, 2] 涉嫌逃票
                    index1 = bboxes.index(suspicion1)
                    index2 = bboxes.index(suspicion2)
                    # print("这两人真实序号：", index1, index2)    # 这两人真实序号： 0 2
                    evade_index_list.append(index1)
                    evade_index_list.append(index2)

                    flag = "WARNING"    # 检出有人逃票，该标识为WARNING
        # 更新每个人的通行状态
        for i in range(len(bboxes)):
            if i in evade_index_list:
                pass_status_list.append(1)    # 出现在涉嫌逃票列表的序号
            else:
                pass_status_list.append(0)
    # 4.更新每个人的track内容：新增闸机编号和通过状态
    for (track, which_gate, pass_status) in zip(tracks, which_gateList, pass_status_list):
        trackContent = TrackContent(gate_num=which_gate,
                                    pass_status=pass_status,
                                    score=track.score,
                                    track_id=track.track_id,
                                    state=track.state,
                                    bbox=Box2Line(track.to_tlbr()),
                                    direction=passway_default_direct_list[which_gate])    # 实时视图用默认方向，后续离线全局视图微调
        TrackContentList.append(trackContent)
    return flag, TrackContentList

'''
    Box转Line：左_上_右_下
'''
def Box2Line(bbox):
    if len(bbox) ==4:
        return "%s_%s_%s_%s" % (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

'''
    判断人在哪个闸机区域，人以中心点为准
    :param bboxes 所有人的人物框列表，左上右下
    :param passway_area_list 通道范围，从左到右，左上右下
    :return which_gates 通道编号列表
'''
def isin_which_gate(bboxes):
    print("bboxes:", bboxes)
    print("passway_area_list:", passway_area_list)
    iou_result = calc_iou(bbox1=bboxes, bbox2=passway_area_list)
    which_gates = []
    for iou in iou_result:
        which_gates.append(iou.argmax())    # 找每行iou最大的，认为该人在该通道内
    return which_gates


'''
    计算iou
'''
def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union