#!/usr/bin/env python3
"""
数据库模式检查脚本
"""

import sqlite3

def check_database_schema():
    """检查数据库表结构"""
    try:
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        
        # 获取所有表名
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = c.fetchall()
        
        print("数据库中的表：")
        for table in tables:
            table_name = table[0]
            print(f"\n📋 表: {table_name}")
            
            # 获取表结构
            c.execute(f"PRAGMA table_info({table_name})")
            columns = c.fetchall()
            
            print("字段信息：")
            for col in columns:
                print(f"  - {col[1]} ({col[2]}) {'PRIMARY KEY' if col[5] else ''}")
        
        conn.close()
        
    except Exception as e:
        print(f"检查数据库时出错：{e}")

if __name__ == "__main__":
    check_database_schema()