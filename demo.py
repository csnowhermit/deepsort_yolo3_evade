import os
import time
from common.config import normal_save_path, evade_save_path, ip
from common.dateUtil import formatTimestamp

flag = "NORMAL"
flag = "WARNING"
read_t1 = time.time()

if flag == "NORMAL":
    savefile = os.path.join(normal_save_path, ip + "_" + formatTimestamp(int(read_t1)) + ".jpg")
else:
    savefile = os.path.join(evade_save_path, ip + "_" + formatTimestamp(int(read_t1)) + ".jpg")

savefile = os.path.join(normal_save_path, ip + "_" + formatTimestamp(int(read_t1)) + ".jpg")
print(savefile)
savefile = os.path.join(evade_save_path, ip + "_" + formatTimestamp(int(read_t1)) + ".jpg")
print(savefile)