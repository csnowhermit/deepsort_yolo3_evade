from common.evadeUtil import calc_iou
from common.config import person_nms_iou

'''
    小孩识别成大人的重叠识别问题，做nms
'''

'''
    手动做一次nms，处理 小孩识别成大人 的情况
    :param person_classes 人的类别
    :param person_boxs 左上右下
    :param person_scores 得分值
'''
def calc_person_nms(person_classes, person_boxs, person_scores):
    # 对人物单独做一次nms
    nms_box_indexes = []    # 被抑制掉的框序号
    for i in range(len(person_boxs)):
        for j in range(i + 1, len(person_boxs)):  # i+1起步，避免跟自身算iou
            box1 = person_boxs[i]
            box2 = person_boxs[j]

            # if person_classes[i] == person_classes[j]:  # 同类不用算这个
            #     continue
            iou_result = calc_iou(box1, box2)
            # if iou_result == 1:
            # print(i, j, iou_result[0])
            if iou_result[0] > 0 and iou_result[0] < 1:  # 如果两个框有交集，且大于阀值
                if iou_result[0] > person_nms_iou:
                    # final = max(person_scores[i], person_scores[j])    # 谁得分大就选定是谁
                    if person_scores[i] > person_scores[j]:
                        final = j
                    else:
                        final = i
                    nms_box_indexes.append(final)
                    # print("now:", nms_box_indexes)
            else:
                pass

    new_person_classes = []
    new_person_scores = []
    new_person_boxs = []
    for i in range(len(person_classes)):
        if i not in nms_box_indexes:
            new_person_classes.append(person_classes[i])
            box = person_boxs[i]
            left, top, right, bottom = box
            new_person_boxs.append([int(left), int(top), int(right - left), int(bottom - top)])      # 转为 左上宽高，供tracker用

            new_person_scores.append(person_scores[i])
    return new_person_classes, new_person_boxs, new_person_scores

if __name__ == '__main__':
    person_classes = ['child', 'head', 'head']
    person_scores = [0.4166157, 0.61927605, 0.9575744]
    person_boxs = [[542.0016, 278.24387, 600.6658, 355.25095], [537.6127, 283.58765, 604.105, 350.898], [546.5691, 78.46593, 627.095, 171.85654]]

    person_classes, person_boxs, person_scores = calc_person_nms(person_classes, person_boxs, person_scores)

    print(person_classes)
    print(person_boxs)
    print(person_scores)