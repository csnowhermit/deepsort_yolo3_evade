#!/usr/local/bin/python3
import zipfile  # 引入zip管理模块
import os
import time

'''
    图像压缩
'''

# 定义一个函数，递归读取absDir文件夹中所有文件，并塞进zipFile文件中。参数absDir表示文件夹的绝对路径。
def writeAllFileToZip(absDir, zipFile):
    i = 0
    for f in os.listdir(absDir):
        i += 1
        absFile = os.path.join(absDir, f)  # 子文件的绝对路径
        zipFile.write(absFile)
        if i % 10000 == 0:
            print("\t processing:", i)
    print("\t processing:", i)
    return 0


if __name__ == '__main__':
    base_path = "D:/monitor_images/10.6.8.181/normal_images/"
    zip_target = "D:/monitor_images/"

    for ddate_dir in os.listdir(base_path):
        src_path = os.path.join(base_path, ddate_dir)
        target_zip = ddate_dir + ".zip"

        target_full_path = os.path.join(zip_target, target_zip)
        print(target_full_path)

        zipFile = zipfile.ZipFile(target_full_path, "w", zipfile.ZIP_DEFLATED)
        # 创建空的zip文件(ZipFile类型)。参数w表示写模式。zipfile.ZIP_DEFLATE表示需要压缩，文件会变小。ZIP_STORED是单纯的复制，文件大小没变。

        start = time.time()
        writeAllFileToZip(src_path, zipFile)  # 开始压缩。如果当前工作目录跟脚本所在目录一样，直接运行这个函数。
        # 执行这条压缩命令前，要保证当前工作目录是脚本所在目录(absDir的父级目录)。否则会报找不到文件的错误。
        print("压缩成功: %s , 耗时: %.3f s" % (ddate_dir, time.time() - start))

        # 压缩成功后将该文件移动到
