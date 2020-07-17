import pymysql
from common.config import *

'''
    数据库操作工具
'''

'''
    批量入库：识别明细数据
'''
def saveManyDetails2DB():
    sql = "INSERT INTO USER1(name, age) VALUES (%s, %s);"
    data = [("Alex", 18), ("Egon", 20), ("Yuan", 21)]
    try:
        # 批量执行多条插入SQL语句
        cursor.executemany(sql, data)
        # 提交事务
        conn.commit()

    except Exception as e:
        # 有异常，回滚事务
        conn.rollback()