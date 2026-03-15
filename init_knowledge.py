import sqlite3
import json
from datetime import datetime

def init_knowledge_base():
    """初始化知识库数据"""
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    # 示例知识库内容
    knowledge_data = [
        {
            "topic": "Python编程基础",
            "content": """
Python是一种高级编程语言，具有简单易学、功能强大的特点。
基本语法包括：
- 变量定义：不需要声明类型，直接赋值
- 数据类型：int, float, string, list, dict, tuple, set
- 控制结构：if-else, for循环, while循环
- 函数定义：使用def关键字
- 类定义：使用class关键字

Python适合数据分析、Web开发、人工智能等领域。
"""
        },
        {
            "topic": "机器学习基础",
            "content": """
机器学习是人工智能的一个分支，让计算机通过数据学习规律。
主要类型包括：
1. 监督学习：有标签数据，如分类、回归
2. 无监督学习：无标签数据，如聚类、降维
3. 强化学习：通过与环境交互学习

常用算法：
- 线性回归、逻辑回归
- 决策树、随机森林
- 支持向量机
- 神经网络
- K-means聚类

应用场景：图像识别、自然语言处理、推荐系统等。
"""
        },
        {
            "topic": "数据库设计",
            "content": """
数据库设计是构建高效数据存储系统的关键。
设计步骤：
1. 需求分析：了解业务需求
2. 概念设计：使用ER图设计实体关系
3. 逻辑设计：转换为关系模式
4. 物理设计：考虑存储和性能

设计原则：
- 规范化：减少数据冗余
- 完整性：保证数据准确性
- 安全性：控制数据访问
- 性能：优化查询速度

常用数据库：MySQL、PostgreSQL、SQLite、MongoDB等。
"""
        },
        {
            "topic": "Web开发基础",
            "content": """
Web开发包括前端和后端两个部分。
前端技术：
- HTML：网页结构
- CSS：样式设计
- JavaScript：交互功能
- 框架：React、Vue、Angular

后端技术：
- 服务器：Apache、Nginx
- 编程语言：Python、Java、Node.js
- 框架：Django、Flask、Spring
- 数据库：MySQL、PostgreSQL

开发流程：
1. 需求分析
2. 架构设计
3. 前后端开发
4. 测试调试
5. 部署上线
"""
        },
        {
            "topic": "数据结构与算法",
            "content": """
数据结构是计算机存储、组织数据的方式。
基本数据结构：
- 线性结构：数组、链表、栈、队列
- 树形结构：二叉树、平衡树、堆
- 图结构：有向图、无向图
- 哈希表：键值对存储

常用算法：
- 排序：冒泡、快速、归并、堆排序
- 搜索：线性搜索、二分搜索
- 图算法：深度优先、广度优先、最短路径
- 动态规划：最优子结构

时间复杂度和空间复杂度是评估算法效率的重要指标。
"""
        },
        {
            "topic": "人工智能伦理",
            "content": """
人工智能伦理关注AI技术的道德和社会影响。
主要议题：
1. 隐私保护：个人数据安全
2. 算法偏见：公平性问题
3. 透明度：可解释性AI
4. 责任归属：AI决策责任
5. 就业影响：自动化与失业
6. 人机关系：AI与人类协作

伦理原则：
- 有益性：AI应该造福人类
- 无害性：避免造成伤害
- 自主性：尊重人类选择权
- 公正性：确保公平对待
- 透明性：可理解和可解释

需要制定相关法律法规和行业标准。
"""
        }
    ]
    
    # 清空现有知识库
    c.execute("DELETE FROM knowledge_base")
    
    # 插入新知识
    for knowledge in knowledge_data:
        c.execute("""
            INSERT INTO knowledge_base (topic, content, created_at) 
            VALUES (?, ?, ?)
        """, (knowledge["topic"], knowledge["content"], datetime.now()))
    
    conn.commit()
    conn.close()
    
    print(f"成功初始化知识库，共添加了 {len(knowledge_data)} 条知识记录")

if __name__ == "__main__":
    init_knowledge_base()