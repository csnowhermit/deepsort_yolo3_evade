import time
import datetime

# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

'''
    格式化时间戳
'''
def formatTimestamp(timestamp, format="%Y-%m-%d_%H:%M:%S"):
    time_tuple = time.localtime(timestamp)
    dt = time.strftime(format, time_tuple)
    return dt

if __name__ == '__main__':
    print(formatTimestamp(int(time.time())))
    print(formatTimestamp(int(time.time()), format="%Y-%m-%d_%H:%M"))