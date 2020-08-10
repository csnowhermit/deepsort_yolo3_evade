from reco_analysis.fileList import fileList
from reco_analysis.typeList import typeList
from common.config import conn, cursor

'''
    计算w1/w2
'''


def getAreaRate(boxList):
    areaList = []
    for box in boxList:
        arr = box.split("_")
        left, top, right, bottom = int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
        areaList.append(abs((abs(right) - abs(left)) * (abs(bottom) - abs(top))))

    minArea = min(areaList)
    maxArea = max(areaList)
    min_box = boxList[areaList.index(minArea)]
    max_box = boxList[areaList.index(maxArea)]
    return maxArea, max_box, minArea, min_box, float(maxArea / minArea)

tableNameList = ['ttt181', 'ttt222']

for tablename in tableNameList:
    sql1 = '''
        select savefile, gate_num, cnt
        from (select savefile, gate_num, count(distinct person_id) cnt 
        from %s 
        group by savefile, gate_num) tmp
        where cnt > 1
        order by cnt desc; 
    ''' % (tablename)

    cursor.execute(sql1)
    results = cursor.fetchall()  # results[0], <class 'tuple'>
    for res in results:
        savefile = res[0]
        gate_num = res[1]

        sql2 = '''
            select savefile, gate_num, person_id, box from %s 
            where savefile='%s' and gate_num=%d
        ''' % (tablename, savefile, int(gate_num))

        cursor.execute(sql2)
        results2 = cursor.fetchall()
        boxList = []
        for res2 in results2:
            # print(res2)
            savefile = res2[0]
            gate_num = int(res2[1])
            person_id = res2[2]
            box = res2[3]
            boxList.append(box)
        # print("==============")
        maxArea, max_box, minArea, min_box, areaRate = getAreaRate(boxList)

        # index = fileList[savefile]
        filename = savefile[43:]
        types = typeList[fileList.index(filename)]

        print(savefile, types, gate_num, maxArea, max_box, minArea, min_box, areaRate)


