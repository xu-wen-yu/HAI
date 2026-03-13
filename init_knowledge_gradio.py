#!/usr/bin/env python3
"""
HAI学伴系统 - 知识库初始化脚本
为系统添加示例知识库内容，增强AI学伴的回答能力
"""

import sqlite3
import json
from datetime import datetime

def init_knowledge_base():
    """初始化知识库内容"""
    print("📚 正在初始化知识库...")
    
    # 示例知识库内容
    knowledge_data = [
        {
            "topic": "数学基础 - 代数概念",
            "content": "代数是数学的一个分支，主要研究数和符号的运算规则。基础代数包括变量、方程、函数等概念。学习代数有助于培养逻辑思维和问题解决能力。建议从简单的一元一次方程开始，逐步学习更复杂的概念。",
            "subject": "数学",
            "difficulty_level": "beginner",
            "tags": "代数,基础,变量,方程,函数"
        },
        {
            "topic": "英语学习 - 语法基础",
            "content": "英语语法是英语学习的重要组成部分。基础语法包括时态、语态、句子结构等。掌握语法有助于提高写作和口语表达能力。建议从简单句开始，逐步学习复合句和复杂句型。",
            "subject": "英语",
            "difficulty_level": "beginner",
            "tags": "语法,时态,句子结构,基础,写作"
        },
        {
            "topic": "物理学习 - 力学基础",
            "content": "力学是物理学的基础分支，研究物体的运动和力的作用。牛顿三大定律是力学的核心内容。理解力学概念有助于解释日常生活中的运动现象。建议通过实验和实例来加深理解。",
            "subject": "物理",
            "difficulty_level": "intermediate",
            "tags": "力学,牛顿定律,运动,力,实验"
        },
        {
            "topic": "学习方法 - 记忆技巧",
            "content": "有效的记忆技巧对于学习非常重要。常用的记忆方法包括联想记忆、重复记忆、图像记忆等。艾宾浩斯遗忘曲线告诉我们及时复习的重要性。建议制定合理的复习计划，结合多种记忆方法。",
            "subject": "学习方法",
            "difficulty_level": "beginner",
            "tags": "记忆,学习方法,复习,艾宾浩斯,技巧"
        },
        {
            "topic": "时间管理 - 学习计划制定",
            "content": "制定合理的学习计划是时间管理的重要技能。SMART原则（具体、可测量、可达成、相关性、时限性）可以帮助制定有效的学习目标。建议将大目标分解为小任务，合理安排学习时间，保持学习的连续性。",
            "subject": "学习方法",
            "difficulty_level": "intermediate",
            "tags": "时间管理,学习计划,SMART原则,目标设定,效率"
        },
        {
            "topic": "数学进阶 - 几何证明",
            "content": "几何证明是数学中的重要技能，需要逻辑推理和空间想象能力。常用的证明方法包括直接证明、反证法、归纳法等。掌握几何证明有助于培养严密的逻辑思维。建议从简单的定理开始，逐步练习复杂的证明题。",
            "subject": "数学",
            "difficulty_level": "advanced",
            "tags": "几何,证明,逻辑推理,定理,空间想象"
        },
        {
            "topic": "英语学习 - 词汇积累策略",
            "content": "词汇是语言学习的基础。有效的词汇积累策略包括词根词缀法、语境记忆法、分类记忆法等。建议每天学习一定数量的新词汇，并在实际语境中使用，通过阅读和写作来巩固词汇记忆。",
            "subject": "英语",
            "difficulty_level": "intermediate",
            "tags": "词汇,词根词缀,语境记忆,积累,阅读写作"
        },
        {
            "topic": "化学学习 - 元素周期表",
            "content": "元素周期表是化学学习的重要工具，展示了元素的原子结构和化学性质的周期性规律。理解周期表有助于预测元素的性质和化学反应。建议掌握周期表的基本结构，理解周期性和族的概念。",
            "subject": "化学",
            "difficulty_level": "beginner",
            "tags": "元素周期表,原子结构,化学性质,周期性,族"
        },
        {
            "topic": "学习心理学 - 注意力集中",
            "content": "注意力集中是学习效果的关键因素。影响注意力的因素包括环境干扰、心理状态、身体健康等。提高注意力的方法包括番茄工作法、冥想练习、环境优化等。建议创造良好的学习环境，合理安排学习时间，适当休息。",
            "subject": "心理学",
            "difficulty_level": "intermediate",
            "tags": "注意力,集中力,番茄工作法,冥想,学习环境"
        },
        {
            "topic": "编程学习 - 算法思维培养",
            "content": "算法思维是编程学习的核心能力，包括问题分解、模式识别、抽象思维等。培养算法思维有助于解决复杂的编程问题。建议从简单的算法开始，如排序、搜索等，逐步学习动态规划、贪心算法等高级算法。",
            "subject": "计算机科学",
            "difficulty_level": "intermediate",
            "tags": "算法,编程思维,问题分解,动态规划,贪心算法"
        }
    ]
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        # 清空现有知识库（可选）
        c.execute("DELETE FROM knowledge_base")
        
        # 插入新知识
        for knowledge in knowledge_data:
            c.execute("""
                INSERT INTO knowledge_base (topic, content)
                VALUES (?, ?)
            """, (
                knowledge["topic"],
                knowledge["content"]
            ))
        
        conn.commit()
        print(f"✅ 成功添加 {len(knowledge_data)} 条知识库内容")
        
    except Exception as e:
        print(f"❌ 知识库初始化失败：{e}")
        conn.rollback()
    finally:
        conn.close()

def add_sample_users():
    """添加示例用户数据"""
    print("👥 正在添加示例用户...")
    
    from werkzeug.security import generate_password_hash
    
    sample_users = [
        {
            "username": "student_demo",
            "password": "demo123",
            "user_type": "free",
            "user_role": "learner",
            "profile": {
                "full_name": "张同学",
                "age": 16,
                "grade_level": "高一",
                "learning_goals": ["提高数学成绩", "掌握英语语法", "培养学习习惯"],
                "preferred_subjects": ["数学", "英语"],
                "learning_style": "visual",
                "weekly_study_hours": 15
            }
        },
        {
            "username": "premium_user",
            "password": "premium123",
            "user_type": "paid",
            "user_role": "learner",
            "profile": {
                "full_name": "李学霸",
                "age": 18,
                "grade_level": "高三",
                "learning_goals": ["高考数学满分", "英语六级通过", "物理竞赛获奖"],
                "preferred_subjects": ["数学", "物理", "化学"],
                "learning_style": "mixed",
                "weekly_study_hours": 25
            }
        },
        {
            "username": "teacher_wang",
            "password": "teacher123",
            "user_type": "paid",
            "user_role": "tutor",
            "profile": None
        }
    ]
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        for user_data in sample_users:
            # 检查用户是否已存在
            c.execute("SELECT id FROM users WHERE username = ?", (user_data["username"],))
            if c.fetchone():
                print(f"⚠️ 用户 {user_data['username']} 已存在，跳过创建")
                continue
            
            # 创建用户
            password_hash = generate_password_hash(user_data["password"])
            c.execute("""
                INSERT INTO users (username, password_hash, user_type)
                VALUES (?, ?, ?)
            """, (
                user_data["username"],
                password_hash,
                user_data["user_type"]
            ))
            
            user_id = c.lastrowid
            
            # 创建学习者档案
            if user_data["profile"] and user_data["user_role"] == "learner":
                profile = user_data["profile"]
                c.execute("""
                    INSERT INTO learner_profiles 
                    (user_id, full_name, age, grade_level, learning_goals, 
                     preferred_subjects, learning_style, weekly_study_hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    profile["full_name"],
                    profile["age"],
                    profile["grade_level"],
                    json.dumps(profile["learning_goals"], ensure_ascii=False),
                    json.dumps(profile["preferred_subjects"], ensure_ascii=False),
                    profile["learning_style"],
                    profile["weekly_study_hours"]
                ))
        
        conn.commit()
        print(f"✅ 成功添加示例用户数据")
        
    except Exception as e:
        print(f"❌ 添加示例用户失败：{e}")
        conn.rollback()
    finally:
        conn.close()

def add_sample_goals_and_progress():
    """添加示例学习目标和学习进度"""
    print("📈 正在添加示例学习数据...")
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        # 获取示例用户ID
        c.execute("SELECT id FROM users WHERE username = 'student_demo'")
        student_id = c.fetchone()[0]
        
        # 示例学习目标
        sample_goals = [
            ("数学期末考试达到90分", "通过系统复习代数、几何、三角函数等内容，在期末考试中取得90分以上的成绩", "2024-06-15", "high"),
            ("英语词汇量提升到5000", "每天学习20个新单词，通过阅读和写作练习巩固记忆", "2024-05-30", "medium"),
            ( "物理力学概念掌握", "深入理解牛顿定律、动量守恒、能量守恒等核心概念", "2024-06-01", "high")
        ]
        
        for title, description, target_date, priority in sample_goals:
            c.execute("""
                INSERT INTO learning_goals 
                (user_id, goal_title, goal_description, target_date, priority, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (student_id, title, description, target_date, priority, "active"))
        
        # 示例学习进度
        sample_progress = [
            ("数学", "一元二次方程", 75, "intermediate", 120),
            ("数学", "三角函数", 60, "beginner", 90),
            ("英语", "被动语态", 85, "intermediate", 60),
            ("物理", "牛顿第一定律", 90, "advanced", 45),
            ("物理", "动量守恒", 70, "intermediate", 80)
        ]
        
        for subject, topic, progress, mastery, study_time in sample_progress:
            c.execute("""
                INSERT INTO learning_progress 
                (user_id, subject, topic, progress_percentage, mastery_level, 
                 study_time_minutes, last_studied)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (student_id, subject, topic, progress, mastery, study_time, datetime.now()))
        
        conn.commit()
        print(f"✅ 成功添加示例学习数据")
        
    except Exception as e:
        print(f"❌ 添加示例学习数据失败：{e}")
        conn.rollback()
    finally:
        conn.close()

def main():
    """主函数"""
    print("🚀 HAI学伴系统 - 知识库初始化")
    print("=" * 50)
    
    try:
        # 初始化知识库
        init_knowledge_base()
        
        # 添加示例用户
        add_sample_users()
        
        # 添加示例学习数据
        add_sample_goals_and_progress()
        
        print("\n🎉 知识库初始化完成！")
        print("💡 您现在可以使用以下测试账户登录：")
        print("   - 学生用户：student_demo / demo123")
        print("   - 付费用户：premium_user / premium123")
        print("   - 教师用户：teacher_wang / teacher123")
        print("\n📖 系统已加载丰富的知识库内容，AI学伴将为您提供更好的学习支持！")
        
    except Exception as e:
        print(f"\n❌ 初始化过程出现错误：{e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())