import os
import shutil

'''
    分析：手机识别成人
'''

targetList = ["大人带小孩",
              '手机识别成人',
              '投影识别成人',
              '抱小孩',
              '未知',
              '肩膀识别成人',
              '腿识别成人',
              '行李识别成人',
              '越闸机',
              '逃票',
              '闸机分组越界',
              '闸机门识别成人'
              ]


file = "D:/workspace/reco.txt"
base_path = "E:/文化公园逃票图片/"

fileList = []
typeList = []
with open(file, encoding='utf-8') as fo:
    for line in fo.readlines():
        line = line.strip("\n")
        arr = line.split(",")

        # if arr[2] != "行李识别成人":
        #     continue

        dest_path = "E:/dest_origin_path/" + arr[2]

        if os.path.exists(dest_path) is False:
            os.makedirs(dest_path)

        filename = arr[1]
        print(filename)
        fileList.append(filename)

        types = arr[2]
        typeList.append(types)

        # arrs = filename.split("_")
        # ip = arrs[0]
        # ddate = arrs[1]
        # print(ip, ddate, dest_path)
        # filepath = base_path + ip + "/evade_origin_images/" + ddate + "/"
        # src  = os.path.join(filepath, filename[0:-4] + "-origin.jpg")
        # if os.path.exists(src) is False:
        #     print("WARNING!!! %s not exists" % src)
        #     continue
        # destfile = os.path.join(dest_path, filename)
        #
        # shutil.copyfile(src, destfile)
print(len(fileList), fileList)
print(len(typeList), typeList)