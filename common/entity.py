import os

'''
    需要用到的实体类
'''

'''
    摄像头点位-物理位置点位对应关系
'''
class CapLocation:
    def __init__(self, gate_num, direction, default_direct, entrance,
                 entrance_direct, entrance_gate_num, displacement,
                 passway_area, gate_area, gate_light_area):
        self.gate_num = gate_num    # 画面中闸机编号
        self.direction = direction    # 方向：0出站，1进站，2双向
        self.default_direct = default_direct    # 默认方向：针对2，跟谁在一起就视为谁的方向
        self.entrance = entrance    # 出入口
        self.entrance_direct = entrance_direct    # 出入口的方向：0出站，1进站
        self.entrance_gate_num = entrance_gate_num    # 出入口的第几个闸机，针对一排闸机用多个摄像头的情况
        self.displacement = displacement    # 画面上是向上走（up）还是向下走（down）
        self.passway_area = getBox(passway_area)    # 目标框的格式转换
        self.gate_area = getBox(gate_area)
        self.gate_light_area = getBox(gate_light_area)

'''
    每一个候选框：左上右下
'''
class Box:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

'''
    box格式转换
'''
def getBox(line):
    if line is None or len(line) < 1:
        return None

    arr = line.split("_")
    if len(arr) == 4:
        return Box(int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]))
    else:
        return None
























