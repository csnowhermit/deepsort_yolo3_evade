import time
import random
from common.dateUtil import formatTimestamp

# print(time.localtime())

# print(formatTimestamp((int(time.time())), format='%Y-%m-%d %H:%M:%S.%f'))

# now = time.time()
# print(time.strftime('%Y-%m-%d %H:%M:%S.%f'))

def formatTimestamp2ms(timestamp, format="%Y-%m-%d_%H:%M:%S"):
    local_time = time.localtime(timestamp)
    data_head = time.strftime(format, local_time)
    data_secs = (timestamp - int(timestamp)) * 1000
    formatted_time_stamp = "%s.%03d" % (data_head, data_secs)
    return formatted_time_stamp

for i in range(100):
    read_t1 = time.time()
    n = 1000 * read_t1
    print(formatTimestamp2ms(time.time()))
    time.sleep(random.uniform(0, 0.1))