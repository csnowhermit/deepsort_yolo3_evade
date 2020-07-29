import math
from tmp.savefileList import savefileList
from common.config import conn, cursor

'''
    分析带小孩的间距
'''

def getBox2(box):
    arr = box.split("_")
    return int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])

real_evade = ['D:/monitor_images/10.6.8.181/evade_images/10.6.8.181_20200727_223438.jpg',
 'D:/monitor_images/10.6.8.181/evade_images/10.6.8.181_20200728_074858.jpg',
 'D:/monitor_images/10.6.8.181/evade_images/10.6.8.181_20200728_124721.jpg',
 'D:/monitor_images/10.6.8.181/evade_images/10.6.8.181_20200728_142632.jpg']

for savefile in savefileList:
    if savefile in real_evade:
        # 1.找到涉嫌逃票的人员列表
        sql = '''
            select person_id from tmp where savefile='%s';
        ''' % (savefile)

        cursor.execute(sql)
        results = cursor.fetchall()
        person_ids = [res[0] for res in results]
        # print(person_ids, type(person_ids))

        # 2.拿到其中的两个人
        max_ids = []
        max_id = max(person_ids)    # 最大的
        # print(max_id)
        # person_ids.__delitem__(person_ids.index(max_id))
        while True:
            if max_id in person_ids:
                person_ids.remove(max_id)
            else:
                break
        # print(type(person_ids))
        max_id2 = max(person_ids)
        # print(max_id, max_id2)
        while True:
            if max_id2 in person_ids:
                person_ids.remove(max_id2)
            else:
                break

        max_ids.append(max_id)
        max_ids.append(max_id2)

        # 通过这两个id，拿到box
        sql2 = '''
            select curr_time, savefile, person_id, box, gate_num from tmp where savefile='%s' and person_id in (%d, %d);
        ''' % (savefile, max_id, max_id2)
        cursor.execute(sql2)
        results2 = cursor.fetchall()
        # for res2 in results2:
        #     print(res2)


        if len(results2) %2 != 0:
            # print("===", savefile, sql2)
            continue

        for i in range(0, len(results2), 2):
            box1 = results2[i][3]
            box2 = results2[i+1][3]

            left1, top1, right1, bottom1 = getBox2(box1)
            left2, top2, right2, bottom2 = getBox2(box2)

            center1 = (abs(left1) + (abs(right1) - abs(left1)) / 2,
                       abs(top1) + (abs(bottom1) - abs(top1)) / 2)

            center2 = (abs(left2) + (abs(right2) - abs(left2)) / 2,
                       abs(top2) + (abs(bottom2) - abs(top2)) / 2)

            distance = math.sqrt(((center1[0] - center2[0]) ** 2) +
                                 ((center1[1] - center2[1]) ** 2))

            if savefile.__contains__("10.6.8.181"):
                rate = float(distance/360.0)
            elif savefile.__contains__("10.6.8.222"):
                rate = float(distance/480.0)

            print(savefile, max_id2, box1, max_id, box2, distance, rate)

