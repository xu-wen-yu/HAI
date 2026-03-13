import gradio as gr
import sqlite3
import hashlib
import secrets
from datetime import datetime
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 数据库初始化函数
def init_db():
    """初始化数据库，创建所有必要的表"""
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    # 用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  user_type TEXT DEFAULT 'free',
                  user_role TEXT DEFAULT 'user',
                  credits INTEGER DEFAULT 0,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # 学习者档案表
    c.execute('''CREATE TABLE IF NOT EXISTS learner_profiles
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER UNIQUE,
                  full_name TEXT,
                  age INTEGER,
                  grade_level TEXT,
                  learning_goals TEXT,
                  preferred_subjects TEXT,
                  learning_style TEXT,
                  weekly_study_hours INTEGER DEFAULT 0,
                  total_study_time INTEGER DEFAULT 0,
                  achievements TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # 学习进度表
    c.execute('''CREATE TABLE IF NOT EXISTS learning_progress
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  subject TEXT,
                  topic TEXT,
                  progress_percentage REAL DEFAULT 0,
                  mastery_level TEXT DEFAULT 'beginner',
                  study_time_minutes INTEGER DEFAULT 0,
                  last_studied TIMESTAMP,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # 学习目标表
    c.execute('''CREATE TABLE IF NOT EXISTS learning_goals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  goal_title TEXT,
                  goal_description TEXT,
                  target_date DATE,
                  priority TEXT DEFAULT 'medium',
                  status TEXT DEFAULT 'active',
                  progress_notes TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # 对话表
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  message TEXT NOT NULL,
                  response TEXT NOT NULL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  is_anonymous BOOLEAN DEFAULT TRUE,
                  subject TEXT,
                  difficulty_level TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # 知识库表（RAG）
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge_base
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  topic TEXT NOT NULL,
                  content TEXT NOT NULL,
                  subject TEXT,
                  difficulty_level TEXT DEFAULT 'beginner',
                  tags TEXT,
                  embedding TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # 功德积分表
    c.execute('''CREATE TABLE IF NOT EXISTS merit_points
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  points INTEGER DEFAULT 0,
                  reason TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # AI学伴分身表 - 新增
    c.execute('''CREATE TABLE IF NOT EXISTS ai_avatars
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  owner_id INTEGER,
                  avatar_name TEXT NOT NULL,
                  personality_description TEXT,
                  expertise_subjects TEXT,
                  teaching_style TEXT,
                  avatar_prompt TEXT,
                  is_public BOOLEAN DEFAULT FALSE,
                  rating REAL DEFAULT 0.0,
                  total_ratings INTEGER DEFAULT 0,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (owner_id) REFERENCES users (id))''')
    
    # AI分身服务记录表 - 新增
    c.execute('''CREATE TABLE IF NOT EXISTS avatar_services
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  avatar_id INTEGER,
                  learner_id INTEGER,
                  service_type TEXT DEFAULT 'chat',
                  session_duration INTEGER DEFAULT 0,
                  rating INTEGER,
                  feedback TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (avatar_id) REFERENCES ai_avatars (id),
                  FOREIGN KEY (learner_id) REFERENCES users (id))''')
    
    # AI分身知识库表 - 新增
    c.execute('''CREATE TABLE IF NOT EXISTS avatar_knowledge
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  avatar_id INTEGER,
                  topic TEXT NOT NULL,
                  content TEXT NOT NULL,
                  subject TEXT,
                  difficulty_level TEXT DEFAULT 'beginner',
                  tags TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (avatar_id) REFERENCES ai_avatars (id))''')
    
    conn.commit()
    conn.close()

# AI学伴类
class HAILearningCompanion:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.knowledge_embeddings = None
        
    def load_knowledge_base(self):
        """加载知识库并创建向量表示"""
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        c.execute("SELECT topic, content FROM knowledge_base")
        knowledge_data = c.fetchall()
        conn.close()
        
        if knowledge_data:
            texts = [f"{topic} {content}" for topic, content in knowledge_data]
            self.knowledge_embeddings = self.vectorizer.fit_transform(texts)
            return knowledge_data
        return []
    
    def find_relevant_knowledge(self, query, top_k=3):
        """使用RAG找到相关知识"""
        if self.knowledge_embeddings is None:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.knowledge_embeddings)[0]
        
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        c.execute("SELECT topic, content FROM knowledge_base")
        knowledge_data = c.fetchall()
        conn.close()
        
        # 获取最相关的知识
        relevant_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_knowledge = [knowledge_data[i] for i in relevant_indices if similarities[i] > 0.1]
        
        return relevant_knowledge
    
    def generate_response(self, user_message, user_type='free'):
        """生成AI学伴的回复"""
        # 查找相关知识
        relevant_knowledge = self.find_relevant_knowledge(user_message)
        
        # 构建提示词
        system_prompt = """你是一个专业的学习伙伴，名字叫"HAI学伴"。你的目标是帮助学习者解决专业问题。
        
规则：
1. 你不能透露自己的AI身份，也不能询问学习者的身份信息
2. 保持匿名性，专注于学习内容
3. 对于免费用户，提供基础但有帮助的回答
4. 对于付费用户，提供更深入、更个性化的指导
5. 使用鼓励和友好的语气
"""
        
        if relevant_knowledge:
            knowledge_context = "\n".join([f"主题: {topic}\n内容: {content}" for topic, content in relevant_knowledge])
            system_prompt += f"\n\n相关知识库内容：\n{knowledge_context}"
        
        # 根据用户类型调整回复深度
        if user_type == 'paid':
            system_prompt += "\n\n付费用户专享：提供更详细、结构化的回答，包含更多例子和深入解释。"
        else:
            system_prompt += "\n\n免费用户：提供简洁但有帮助的回答。"
        
        # 模拟AI回复（实际使用时替换为真实的OpenAI API调用）
        response = self.simulate_ai_response(user_message, system_prompt)
        return response
    
    def simulate_ai_response(self, user_message, system_prompt):
        """模拟AI回复（演示用）"""
        responses = [
            f"这是一个很好的问题！关于'{user_message}'，让我来帮你解答。",
            f"我理解你的疑问。'{user_message}'这个问题很有深度，我来详细解释一下。",
            f"很高兴能帮助你学习！关于'{user_message}'，这里是我的建议。"
        ]
        
        # 根据知识库内容增强回复
        if "知识库" in system_prompt:
            return f"基于我的知识库，我找到了相关的信息来帮助你理解'{user_message}'。让我给你一个全面的解答。"
        
        return responses[hash(user_message) % len(responses)]

# AI学伴分身类 - 新增
class AIAvatar:
    def __init__(self, avatar_id, owner_id, avatar_name, personality_description, 
                 expertise_subjects, teaching_style, avatar_prompt):
        self.avatar_id = avatar_id
        self.owner_id = owner_id
        self.avatar_name = avatar_name
        self.personality_description = personality_description
        self.expertise_subjects = expertise_subjects
        self.teaching_style = teaching_style
        self.avatar_prompt = avatar_prompt
        self.vectorizer = TfidfVectorizer()
        self.knowledge_embeddings = None
        
    def load_avatar_knowledge(self):
        """加载分身的专属知识库"""
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        c.execute("SELECT topic, content FROM avatar_knowledge WHERE avatar_id = ?", (self.avatar_id,))
        knowledge_data = c.fetchall()
        conn.close()
        
        if knowledge_data:
            texts = [f"{topic} {content}" for topic, content in knowledge_data]
            self.knowledge_embeddings = self.vectorizer.fit_transform(texts)
            return knowledge_data
        return []
    
    def find_relevant_knowledge(self, query, top_k=3):
        """使用RAG找到分身的相关知识"""
        if self.knowledge_embeddings is None:
            # 如果没有分身知识库，使用通用知识库
            return self.load_general_knowledge()
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.knowledge_embeddings)[0]
        
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        c.execute("SELECT topic, content FROM avatar_knowledge WHERE avatar_id = ?", (self.avatar_id,))
        knowledge_data = c.fetchall()
        conn.close()
        
        # 获取最相关的知识
        relevant_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_knowledge = [knowledge_data[i] for i in relevant_indices if similarities[i] > 0.1]
        
        return relevant_knowledge
    
    def load_general_knowledge(self):
        """加载通用知识库"""
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        c.execute("SELECT topic, content FROM knowledge_base")
        knowledge_data = c.fetchall()
        conn.close()
        return knowledge_data
    
    def generate_response(self, user_message, learner_profile=None):
        """生成分身的个性化回复"""
        # 查找相关知识
        relevant_knowledge = self.find_relevant_knowledge(user_message)
        
        # 构建分身的个性化提示词
        system_prompt = f"""你是"{self.avatar_name}"，一个由真人学伴创建的AI分身。
        
个人特色：
{self.personality_description}

专业领域：{', '.join(self.expertise_subjects)}

教学风格：{self.teaching_style}

{self.avatar_prompt}

规则：
1. 保持你独特的个性和教学风格
2. 根据学习者的背景和需求调整回答
3. 在你的专业领域内提供深度指导
4. 使用鼓励性和个性化的语气
"""
        
        # 根据学习者档案个性化回复
        if learner_profile:
            system_prompt += f"\n\n学习者信息：\n年龄：{learner_profile.get('age', '未知')}\n年级：{learner_profile.get('grade_level', '未知')}\n学习目标：{learner_profile.get('learning_goals', '未设定')}\n偏好学科：{learner_profile.get('preferred_subjects', '未设定')}"
        
        if relevant_knowledge:
            knowledge_context = "\n".join([f"主题: {topic}\n内容: {content}" for topic, content in relevant_knowledge])
            system_prompt += f"\n\n相关知识内容：\n{knowledge_context}"
        
        # 模拟分身回复
        response = self.simulate_avatar_response(user_message, system_prompt)
        return response
    
    def simulate_avatar_response(self, user_message, system_prompt):
        """模拟分身回复（演示用）"""
        # 根据分身个性生成不同风格的回复
        if "幽默" in self.personality_description or "风趣" in self.personality_description:
            return f"哈哈，{user_message}这个问题真有趣！作为{self.avatar_name}，我要用我的{self.teaching_style}方式来帮你解答～"
        elif "严谨" in self.personality_description or "专业" in self.personality_description:
            return f"关于{user_message}，作为专注于{', '.join(self.expertise_subjects)}的{self.avatar_name}，我会用{self.teaching_style}的方法为你详细分析。"
        elif "温柔" in self.personality_description or "耐心" in self.personality_description:
            return f"亲爱的学习者，关于{user_message}，让{self.avatar_name}用{self.teaching_style}的方式来陪伴你一起学习吧～"
        else:
            return f"你好！我是{self.avatar_name}，让我用我的{self.teaching_style}风格来帮你解答{user_message}这个问题。"

# 全局变量
ai_companion = HAILearningCompanion()
current_user = None
active_avatars = {}  # 存储活跃的分身实例

# 用户管理函数
def register_user(username, password, user_type="free", user_role="user", 
                  full_name="", age=None, grade_level="", learning_goals=None, 
                  preferred_subjects=None, learning_style="", weekly_study_hours=0):
    """注册新用户"""
    if not username or not password:
        return "错误：用户名和密码不能为空"
    
    if learning_goals is None:
        learning_goals = []
    if preferred_subjects is None:
        preferred_subjects = []
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        password_hash = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password_hash, user_type, user_role) VALUES (?, ?, ?, ?)", 
                  (username, password_hash, user_type, user_role))
        
        user_id = c.lastrowid
        
        # 如果是学习者角色，创建学习者档案
        if user_role == 'learner':
            c.execute("""
                INSERT INTO learner_profiles 
                (user_id, full_name, age, grade_level, learning_goals, 
                 preferred_subjects, learning_style, weekly_study_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                full_name or username,
                age,
                grade_level,
                json.dumps(learning_goals),
                json.dumps(preferred_subjects),
                learning_style,
                weekly_study_hours
            ))
        
        conn.commit()
        conn.close()
        return f"注册成功！欢迎 {username} 加入HAI学伴系统！"
    except sqlite3.IntegrityError:
        conn.close()
        return "错误：用户名已存在，请选择其他用户名。"
    except Exception as e:
        conn.close()
        return f"注册失败：{str(e)}"

def login_user(username, password):
    """用户登录"""
    global current_user
    
    if not username or not password:
        return "错误：用户名和密码不能为空", None
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    c.execute("SELECT id, password_hash, user_type, user_role FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[1], password):
        current_user = {
            'id': user[0],
            'username': username,
            'user_type': user[2],
            'user_role': user[3]
        }
        return f"登录成功！欢迎回来，{username}！", current_user
    
    return "错误：用户名或密码错误", None

def get_user_profile(user_id):
    """获取用户档案信息"""
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    # 获取用户基本信息
    c.execute("SELECT username, user_type, user_role, credits, created_at FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    
    if not user:
        conn.close()
        return None
    
    # 获取学习者档案
    learner_profile = None
    if user[2] == 'learner':
        c.execute("""
            SELECT full_name, age, grade_level, learning_goals, 
                   preferred_subjects, learning_style, weekly_study_hours, total_study_time
            FROM learner_profiles WHERE user_id = ?
        """, (user_id,))
        learner_data = c.fetchone()
        if learner_data:
            learner_profile = {
                'full_name': learner_data[0],
                'age': learner_data[1],
                'grade_level': learner_data[2],
                'learning_goals': json.loads(learner_data[3]) if learner_data[3] else [],
                'preferred_subjects': json.loads(learner_data[4]) if learner_data[4] else [],
                'learning_style': learner_data[5],
                'weekly_study_hours': learner_data[6],
                'total_study_time': learner_data[7]
            }
    
    # 获取学习进度
    c.execute("""
        SELECT subject, topic, progress_percentage, mastery_level, study_time_minutes, last_studied
        FROM learning_progress WHERE user_id = ?
        ORDER BY last_studied DESC
    """, (user_id,))
    learning_progress = []
    for row in c.fetchall():
        learning_progress.append({
            'subject': row[0],
            'topic': row[1],
            'progress_percentage': row[2],
            'mastery_level': row[3],
            'study_time_minutes': row[4],
            'last_studied': row[5]
        })
    
    # 获取学习目标
    c.execute("""
        SELECT goal_title, goal_description, target_date, priority, status, progress_notes
        FROM learning_goals WHERE user_id = ?
        ORDER BY target_date ASC
    """, (user_id,))
    learning_goals = []
    for row in c.fetchall():
        learning_goals.append({
            'goal_title': row[0],
            'goal_description': row[1],
            'target_date': row[2],
            'priority': row[3],
            'status': row[4],
            'progress_notes': row[5]
        })
    
    conn.close()
    
    return {
        'username': user[0],
        'user_type': user[1],
        'user_role': user[2],
        'credits': user[3],
        'created_at': user[4],
        'learner_profile': learner_profile,
        'learning_progress': learning_progress,
        'learning_goals': learning_goals
    }

# AI对话函数
def chat_with_ai(message, user_type="free"):
    """与AI学伴对话"""
    if not message or not message.strip():
        return "请输入您的问题..."
    
    try:
        # 生成AI回复
        response = ai_companion.generate_response(message, user_type)
        
        # 保存对话记录（如果有当前用户）
        if current_user:
            conn = sqlite3.connect('hai_learn.db')
            c = conn.cursor()
            c.execute("INSERT INTO conversations (user_id, message, response) VALUES (?, ?, ?)",
                      (current_user['id'], message, response))
            
            # 免费用户获得功德积分
            if current_user['user_type'] == 'free':
                c.execute("UPDATE users SET credits = credits + 1 WHERE id = ?", (current_user['id'],))
                c.execute("INSERT INTO merit_points (user_id, points, reason) VALUES (?, ?, ?)",
                          (current_user['id'], 1, "免费帮助学习者"))
            
            conn.commit()
            conn.close()
        
        return response
    except Exception as e:
        return f"抱歉，处理您的问题时出现错误：{str(e)}"

def get_chat_history(user_id, limit=20):
    """获取聊天历史"""
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    c.execute("""
        SELECT message, response, timestamp 
        FROM conversations 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (user_id, limit))
    conversations = c.fetchall()
    conn.close()
    
    history = []
    for msg, resp, timestamp in conversations:
        history.append({
            'message': msg,
            'response': resp,
            'timestamp': timestamp
        })
    
    return history

# 学习管理函数
def add_learning_goal(user_id, goal_title, goal_description, target_date, priority="medium"):
    """添加学习目标"""
    if not goal_title or not target_date:
        return "错误：目标标题和目标日期不能为空"
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        c.execute("""
            INSERT INTO learning_goals 
            (user_id, goal_title, goal_description, target_date, priority, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, goal_title, goal_description, target_date, priority, 'active'))
        
        conn.commit()
        conn.close()
        return "学习目标添加成功！"
    except Exception as e:
        conn.close()
        return f"添加学习目标失败：{str(e)}"

def update_learning_progress(user_id, subject, topic, progress_percentage, mastery_level="beginner", study_time_minutes=0):
    """更新学习进度"""
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        # 检查是否已存在该学习进度
        c.execute("""
            SELECT id FROM learning_progress 
            WHERE user_id = ? AND subject = ? AND topic = ?
        """, (user_id, subject, topic))
        existing = c.fetchone()
        
        if existing:
            # 更新现有进度
            c.execute("""
                UPDATE learning_progress 
                SET progress_percentage = ?, mastery_level = ?, 
                    study_time_minutes = study_time_minutes + ?, 
                    last_studied = ?
                WHERE user_id = ? AND subject = ? AND topic = ?
            """, (progress_percentage, mastery_level, study_time_minutes, 
                  datetime.now(), user_id, subject, topic))
        else:
            # 创建新进度
            c.execute("""
                INSERT INTO learning_progress 
                (user_id, subject, topic, progress_percentage, mastery_level, 
                 study_time_minutes, last_studied)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, subject, topic, progress_percentage, mastery_level, 
                  study_time_minutes, datetime.now()))
        
        conn.commit()
        conn.close()
        return "学习进度更新成功！"
    except Exception as e:
        conn.close()
        return f"更新学习进度失败：{str(e)}"

# 数据可视化函数
def create_learning_dashboard(user_id):
    """创建学习仪表板"""
    conn = sqlite3.connect('hai_learn.db')
    
    # 获取学习进度数据
    progress_df = pd.read_sql_query("""
        SELECT subject, topic, progress_percentage, mastery_level, study_time_minutes, last_studied
        FROM learning_progress 
        WHERE user_id = ?
        ORDER BY last_studied DESC
    """, conn, params=(user_id,))
    
    # 获取学习目标数据
    goals_df = pd.read_sql_query("""
        SELECT goal_title, target_date, priority, status
        FROM learning_goals 
        WHERE user_id = ?
        ORDER BY target_date ASC
    """, conn, params=(user_id,))
    
    conn.close()
    
    if progress_df.empty and goals_df.empty:
        return "您还没有学习记录，开始您的学习之旅吧！"
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('学习进度概览', '学科分布', '目标优先级分布', '学习时长统计'),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    if not progress_df.empty:
        # 学习进度条形图
        fig.add_trace(
            go.Bar(x=progress_df['subject'], y=progress_df['progress_percentage'],
                   name="学习进度", marker_color='lightblue'),
            row=1, col=1
        )
        
        # 学科分布饼图
        subject_counts = progress_df['subject'].value_counts()
        fig.add_trace(
            go.Pie(labels=subject_counts.index, values=subject_counts.values,
                   name="学科分布"),
            row=1, col=2
        )
        
        # 学习时长统计
        fig.add_trace(
            go.Bar(x=progress_df['subject'], y=progress_df['study_time_minutes'],
                   name="学习时长(分钟)", marker_color='lightgreen'),
            row=2, col=2
        )
    
    if not goals_df.empty:
        # 目标优先级分布
        priority_counts = goals_df['priority'].value_counts()
        fig.add_trace(
            go.Bar(x=priority_counts.index, y=priority_counts.values,
                   name="目标优先级", marker_color='orange'),
            row=2, col=1
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="学习仪表板")
    
    return fig

# 知识库管理函数
def add_knowledge_to_base(topic, content, subject="", difficulty_level="beginner", tags=""):
    """添加知识到知识库"""
    if not topic or not content:
        return "错误：知识主题和内容不能为空"
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        c.execute("""
            INSERT INTO knowledge_base 
            (topic, content, subject, difficulty_level, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (topic, content, subject, difficulty_level, tags))
        
        conn.commit()
        conn.close()
        
        # 重新加载知识库
        ai_companion.load_knowledge_base()
        return "知识添加成功！"
    except Exception as e:
        conn.close()
        return f"添加知识失败：{str(e)}"

def search_knowledge_base(query):
    """搜索知识库"""
    if not query:
        return []
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    # 使用LIKE进行模糊搜索
    c.execute("""
        SELECT topic, content, subject, difficulty_level, tags, created_at
        FROM knowledge_base 
        WHERE topic LIKE ? OR content LIKE ? OR subject LIKE ?
        ORDER BY created_at DESC
        LIMIT 10
    """, (f'%{query}%', f'%{query}%', f'%{query}%'))
    
    results = c.fetchall()
    conn.close()
    
    knowledge_list = []
    for topic, content, subject, difficulty, tags, created in results:
        knowledge_list.append({
            'topic': topic,
            'content': content,
            'subject': subject,
            'difficulty': difficulty,
            'tags': tags,
            'created': created
        })
    
    return knowledge_list

# 初始化数据库和AI学伴
init_db()
ai_companion.load_knowledge_base()

# AI分身管理函数 - 新增
def create_ai_avatar(owner_id, avatar_name, personality_description, expertise_subjects, teaching_style, avatar_prompt):
    """创建AI分身"""
    try:
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        
        # 创建分身记录
        c.execute("""INSERT INTO ai_avatars (owner_id, avatar_name, personality_description, expertise_subjects, 
                      teaching_style, avatar_prompt, is_public, created_at, updated_at)
                      VALUES (?, ?, ?, ?, ?, ?, 0, datetime('now'), datetime('now'))""",
                 (owner_id, avatar_name, personality_description, json.dumps(expertise_subjects), 
                  teaching_style, avatar_prompt))
        
        avatar_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return f"✅ AI分身 '{avatar_name}' 创建成功！ID: {avatar_id}"
    except Exception as e:
        return f"❌ 创建分身失败: {str(e)}"

def get_user_avatars(owner_id):
    """获取用户的所有分身"""
    try:
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        c.execute("""SELECT id, avatar_name, personality_description, expertise_subjects, 
                    teaching_style, is_public FROM ai_avatars WHERE owner_id = ?""", (owner_id,))
        avatars = c.fetchall()
        conn.close()
        
        return avatars
    except Exception as e:
        print(f"获取用户分身失败: {e}")
        return []

def get_learner_profile(user_id):
    """获取学习者档案"""
    try:
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        c.execute("""SELECT age, grade_level, learning_goals, preferred_subjects, 
                    learning_style, weekly_study_hours FROM learner_profiles WHERE user_id = ?""", (user_id,))
        profile_data = c.fetchone()
        conn.close()
        
        if profile_data:
            return {
                'age': profile_data[0],
                'grade_level': profile_data[1],
                'learning_goals': json.loads(profile_data[2]) if profile_data[2] else [],
                'preferred_subjects': json.loads(profile_data[3]) if profile_data[3] else [],
                'learning_style': profile_data[4],
                'weekly_study_hours': profile_data[5],
                'user_id': user_id
            }
        return None
    except Exception as e:
        print(f"获取学习者档案失败: {e}")
        return None

def activate_avatar(avatar_id):
    """激活分身实例"""
    try:
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        c.execute("""SELECT avatar_name, personality_description, expertise_subjects, 
                    teaching_style, avatar_prompt FROM ai_avatars WHERE id = ?""", 
                 (avatar_id,))
        avatar_data = c.fetchone()
        conn.close()
        
        if avatar_data:
            avatar_name, personality_description, expertise_subjects, teaching_style, avatar_prompt = avatar_data
            expertise_list = json.loads(expertise_subjects)
            
            # 创建分身实例
            avatar = AIAvatar(avatar_id, avatar_data[0], avatar_name, personality_description, 
                            expertise_list, teaching_style, avatar_prompt)
            active_avatars[avatar_id] = avatar
            
            return f"✅ 分身 '{avatar_name}' 已激活！"
        else:
            return "❌ 分身不存在"
    except Exception as e:
        return f"❌ 激活分身失败: {str(e)}"

def chat_with_avatar(avatar_id, user_message, learner_profile=None):
    """与分身对话"""
    try:
        # 检查分身是否已激活
        if avatar_id not in active_avatars:
            result = activate_avatar(avatar_id)
            if "❌" in result:
                return result
        
        avatar = active_avatars[avatar_id]
        response = avatar.generate_response(user_message, learner_profile)
        
        # 记录对话历史
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        c.execute("""INSERT INTO avatar_services (avatar_id, learner_id, service_type, 
                      session_duration, rating, feedback, timestamp)
                      VALUES (?, ?, 'chat', 0, NULL, NULL, datetime('now'))""",
                 (avatar_id, learner_profile.get('user_id') if learner_profile else None))
        conn.commit()
        conn.close()
        
        return response
    except Exception as e:
        return f"❌ 与分身对话失败: {str(e)}"

def add_avatar_knowledge(avatar_id, topic, content):
    """为分身添加专属知识"""
    try:
        conn = sqlite3.connect('hai_learn.db')
        c = conn.cursor()
        c.execute("""INSERT INTO avatar_knowledge (avatar_id, topic, content, subject, difficulty_level, tags, created_at)
                      VALUES (?, ?, ?, '', 'beginner', '', datetime('now'))""", (avatar_id, topic, content))
        conn.commit()
        conn.close()
        
        # 重新加载分身的知识库
        if avatar_id in active_avatars:
            active_avatars[avatar_id].load_avatar_knowledge()
        
        return f"✅ 已为分身添加知识：{topic}"
    except Exception as e:
        return f"❌ 添加分身知识失败: {str(e)}"

# Gradio界面函数
def create_gradio_interface():
    """创建Gradio界面"""
    
    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .header-text {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .welcome-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .welcome-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    .btn-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    .btn-secondary {
        background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(149, 165, 166, 0.3) !important;
    }
    .btn-secondary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(149, 165, 166, 0.4) !important;
    }
    .status-info {
        background: linear-gradient(135deg, #e8f4f8 0%, #d1ecf1 100%) !important;
        border: 1px solid #bee5eb !important;
        color: #0c5460 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        padding: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
    .status-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        color: #2d3748 !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
    .status-error {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        color: #2d3748 !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
    .data-table {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        background: white !important;
    }
    .chat-output {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 12px !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        line-height: 1.6 !important;
        padding: 15px !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    .stats-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    /* 标签页样式 */
    .tab-nav {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 12px !important;
        padding: 8px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    .tab-nav button {
        background: transparent !important;
        border: none !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        margin: 0 2px !important;
    }
    .tab-nav button.selected {
        background: rgba(255,255,255,0.2) !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    /* 输入框样式 */
    input[type="text"], textarea, select {
        border-radius: 10px !important;
        border: 2px solid #e9ecef !important;
        transition: all 0.3s ease !important;
        background: white !important;
        padding: 12px !important;
        font-size: 14px !important;
    }
    input[type="text"]:focus, textarea:focus, select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
        outline: none !important;
        transform: translateY(-1px) !important;
    }
    /* 按钮动画效果 */
    button {
        transition: all 0.3s ease !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        border: none !important;
        cursor: pointer !important;
    }
    button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
    }
    button:active {
        transform: translateY(0) !important;
    }
    /* 分组样式 */
    .gr-group {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%) !important;
        border-radius: 15px !important;
        padding: 25px !important;
        margin: 15px 0 !important;
        border: 1px solid #e9ecef !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        backdrop-filter: blur(10px) !important;
    }
    /* 滑块样式 */
    .gr-slider input[type="range"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px !important;
        height: 6px !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    }
    /* 单选按钮样式 */
    .gr-radio input[type="radio"]:checked + label {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: #667eea !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    }
    /* 整体卡片效果 */
    .gr-box {
        border-radius: 15px !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    .gr-box:hover {
        box-shadow: 0 12px 35px rgba(0,0,0,0.12) !important;
        transform: translateY(-2px) !important;
    }
    """
    
    app_theme = gr.themes.Soft()

    with gr.Blocks(title="HAI学伴系统 - AI学习伙伴") as app:
        gr.Markdown("""
        <div class="header-text">
            <h1>🤖 HAI学伴系统</h1>
            <h3>您的专属AI学习伙伴</h3>
            <p style="font-size: 16px; margin-top: 10px;">"愿人人皆有学伴，处处皆有智慧"</p>
        </div>
        """)
        
        # 状态变量
        user_state = gr.State(None)
        
        with gr.Tab("🏠 主页"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>🎯 系统特色</h3>
                        <div class="feature-card">
                            <strong>🤖 AI智能对话</strong><br>
                            24小时在线的学习伙伴，随时解答您的问题
                        </div>
                        <div class="feature-card">
                            <strong>🎨 个性化学习</strong><br>
                            根据您的学习风格定制专属内容
                        </div>
                        <div class="feature-card">
                            <strong>📊 进度跟踪</strong><br>
                            可视化学习进展，让进步一目了然
                        </div>
                        <div class="feature-card">
                            <strong>🎯 目标管理</strong><br>
                            设定和追踪学习目标，保持学习动力
                        </div>
                        <div class="feature-card">
                            <strong>📚 知识库RAG</strong><br>
                            基于知识库的智能回答，提供准确信息
                        </div>
                    </div>
                    
                    <div class="welcome-card">
                        <h3>🌟 使用指南</h3>
                        <div style="background: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
                            <ol style="margin: 0; padding-left: 20px; color: #4a5568;">
                                <li>注册并登录您的账户</li>
                                <li>完善学习者档案信息</li>
                                <li>设置学习目标</li>
                                <li>开始与AI学伴对话</li>
                                <li>跟踪学习进度</li>
                            </ol>
                        </div>
                    </div>
                    """)
                
                with gr.Column(scale=2):
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>💬 快速开始</h3>
                        <p style="color: #4a5568; margin-bottom: 20px;">点击下方按钮开始您的学习之旅！</p>
                    </div>
                    """)
                    
                    # 快速登录区域
                    with gr.Group():
                        gr.Markdown("#### 🔐 快速登录")
                        with gr.Row():
                            quick_username = gr.Textbox(label="用户名", placeholder="请输入用户名", scale=1)
                            quick_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码", scale=1)
                        with gr.Row():
                            quick_login_btn = gr.Button("🚀 登录", variant="primary", scale=1)
                            gr.Button("👤 注册新用户", scale=1, elem_classes=["btn-secondary"])
                        quick_login_status = gr.Textbox(label="登录状态", interactive=False, elem_classes=["status-info"])
        
        with gr.Tab("👤 用户管理"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>📝 用户注册</h3>
                        <p style="color: #4a5568;">创建您的专属学习账户</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        gr.Markdown("#### 🔑 账户信息")
                        reg_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                        reg_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                        
                        gr.Markdown("#### 👥 用户类型")
                        with gr.Row():
                            reg_user_type = gr.Radio(choices=["free", "paid"], value="free", label="用户类型", scale=1)
                            reg_user_role = gr.Radio(choices=["user", "learner", "tutor"], value="user", label="用户角色", scale=1)
                    
                    # 学习者档案字段
                    with gr.Group(visible=False) as learner_fields:
                        gr.Markdown("#### 🎓 学习者档案")
                        with gr.Row():
                            reg_full_name = gr.Textbox(label="姓名", placeholder="请输入真实姓名", scale=1)
                            reg_age = gr.Number(label="年龄", minimum=1, maximum=120, scale=1)
                        reg_grade_level = gr.Textbox(label="年级", placeholder="如：高一、大二等")
                        reg_learning_goals = gr.Textbox(label="学习目标", placeholder="用逗号分隔多个目标")
                        reg_preferred_subjects = gr.Textbox(label="偏好学科", placeholder="用逗号分隔多个学科")
                        reg_learning_style = gr.Radio(choices=["visual", "auditory", "kinesthetic", "mixed"], 
                                                    value="mixed", label="学习风格")
                        reg_weekly_study_hours = gr.Number(label="每周学习时间(小时)", minimum=0, maximum=168)
                    
                    reg_btn = gr.Button("🎯 创建账户", variant="primary")
                    reg_status = gr.Textbox(label="注册结果", interactive=False)
                
                with gr.Column():
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>🔐 用户登录</h3>
                        <p style="color: #4a5568;">登录您的学习账户</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                        login_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                        login_btn = gr.Button("🚀 登录", variant="primary")
                        login_status = gr.Textbox(label="登录状态", interactive=False)
                    
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>👤 当前用户</h3>
                    </div>
                    """)
                    
                    current_user_info = gr.Textbox(label="用户信息", interactive=False, lines=5)
                    with gr.Row():
                        logout_btn = gr.Button("🚪 登出", variant="secondary", scale=1)
                        refresh_user_btn = gr.Button("🔄 刷新信息", variant="secondary", scale=1)
                    logout_status = gr.Textbox(label="登出状态", interactive=False)
        
        with gr.Tab("💬 AI学伴"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>🎓 与AI学伴对话</h3>
                        <p style="color: #4a5568;">向您的专属AI学习伙伴提问</p>
                    </div>
                    """)
                    
                    chat_input = gr.Textbox(
                        label="💭 您的问题", 
                        placeholder="请输入您想询问的学习问题...\n例如：如何学好数学？英语语法有哪些重点？", 
                        lines=4,
                        max_lines=6
                    )
                    
                    with gr.Row():
                        chat_btn = gr.Button("🚀 发送问题", variant="primary", scale=2)
                        clear_chat_btn = gr.Button("🗑️ 清空", variant="secondary", scale=1)
                    
                    chat_output = gr.Textbox(
                        label="✨ AI学伴回复", 
                        interactive=False, 
                        lines=6,
                        max_lines=10,
                        elem_classes=["chat-response"]
                    )
                    
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h4>📝 知识库增强</h4>
                        <p style="color: #4a5568; font-size: 14px;">基于知识库的智能回答，提供更准确的学习建议</p>
                    </div>
                    """)
                    
                    with gr.Row():
                        knowledge_query = gr.Textbox(
                            label="🔍 搜索知识库", 
                            placeholder="输入关键词搜索知识库，如：数学、英语、物理...",
                            scale=2
                        )
                        knowledge_search_btn = gr.Button("🔎 搜索", variant="secondary", scale=1)
                    
                    knowledge_results = gr.JSON(
                        label="📊 搜索结果", 
                        height=250,
                        elem_classes=["knowledge-results"]
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>📚 最近对话</h3>
                        <p style="color: #4a5568; font-size: 14px;">查看您的对话历史记录</p>
                    </div>
                    """)
                    
                    chat_history = gr.JSON(
                        label="💬 对话历史", 
                        height=350,
                        elem_classes=["chat-history"]
                    )
                    
                    with gr.Row():
                        refresh_history_btn = gr.Button("🔄 刷新历史", variant="secondary", scale=1)
                        export_chat_btn = gr.Button("📥 导出", variant="secondary", scale=1)
        
        with gr.Tab("📊 学习管理"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>🎯 学习目标管理</h3>
                        <p style="color: #4a5568;">设定和管理您的学习目标</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        gr.Markdown("#### 📝 目标信息")
                        goal_title = gr.Textbox(
                            label="目标标题", 
                            placeholder="请输入学习目标标题，如：掌握Python编程"
                        )
                        goal_description = gr.Textbox(
                            label="目标描述", 
                            placeholder="请详细描述您的学习目标，包括具体内容和预期成果", 
                            lines=3
                        )
                        
                        gr.Markdown("#### ⏰ 时间安排")
                        with gr.Row():
                            goal_target_date = gr.Textbox(
                                label="目标日期", 
                                placeholder="格式：YYYY-MM-DD",
                                scale=1
                            )
                            goal_priority = gr.Radio(
                                choices=["low", "medium", "high"], 
                                value="medium", 
                                label="优先级",
                                scale=1
                            )
                    
                    add_goal_btn = gr.Button("🎯 添加学习目标", variant="primary")
                    goal_status = gr.Textbox(label="操作状态", interactive=False)
                    
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>📈 学习进度更新</h3>
                        <p style="color: #4a5568;">记录您的学习进展</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        gr.Markdown("#### 📚 学习内容")
                        with gr.Row():
                            progress_subject = gr.Textbox(
                                label="学科", 
                                placeholder="如：数学、英语、编程等",
                                scale=1
                            )
                            progress_topic = gr.Textbox(
                                label="主题", 
                                placeholder="如：代数、语法、函数等",
                                scale=1
                            )
                        
                        gr.Markdown("#### 📊 进度信息")
                        with gr.Row():
                            progress_percentage = gr.Slider(
                                label="进度百分比", 
                                minimum=0, 
                                maximum=100, 
                                value=0,
                                scale=2
                            )
                            progress_mastery = gr.Radio(
                                choices=["beginner", "intermediate", "advanced"], 
                                value="beginner", 
                                label="掌握程度",
                                scale=1
                            )
                        
                        progress_study_time = gr.Number(
                            label="本次学习时间(分钟)", 
                            minimum=0,
                            info="记录本次学习花费的时间"
                        )
                    
                    update_progress_btn = gr.Button("📊 更新进度", variant="primary")
                    progress_status = gr.Textbox(label="操作状态", interactive=False)
                
                with gr.Column():
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>📊 学习仪表板</h3>
                        <p style="color: #4a5568;">可视化展示您的学习数据</p>
                    </div>
                    """)
                    
                    dashboard_btn = gr.Button("📈 生成学习仪表板", variant="primary")
                    learning_dashboard = gr.Plot(
                        label="📊 学习数据可视化"
                    )
                    
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h4>💡 使用提示</h4>
                        <div style="background: #f7fafc; padding: 15px; border-radius: 8px; font-size: 14px; color: #4a5568;">
                            <ul style="margin: 0; padding-left: 20px;">
                                <li>定期更新学习进度，保持数据的准确性</li>
                                <li>设置具体可达的学习目标</li>
                                <li>记录每次学习的时间投入</li>
                                <li>根据仪表板调整学习计划</li>
                            </ul>
                        </div>
                    </div>
                    """)
        
        with gr.Tab("📖 知识库管理"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>📚 知识条目管理</h3>
                        <p style="color: #4a5568;">添加和管理学习知识</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        gr.Markdown("#### 📝 基本信息")
                        knowledge_topic = gr.Textbox(
                            label="知识主题", 
                            placeholder="请输入知识主题，如：Python函数定义"
                        )
                        knowledge_subject = gr.Textbox(
                            label="所属学科", 
                            placeholder="如：数学、物理、编程等",
                            info="选择正确的分类有助于更好的组织知识"
                        )
                        knowledge_difficulty = gr.Radio(
                            choices=["beginner", "intermediate", "advanced"], 
                            value="beginner", 
                            label="难度等级",
                            info="选择适合的知识难度等级"
                        )
                        knowledge_tags = gr.Textbox(
                            label="标签", 
                            placeholder="多个标签用逗号分隔，如：基础,重要,易错",
                            info="标签可以帮助快速检索相关内容"
                        )
                        
                        gr.Markdown("#### 📖 知识内容")
                        knowledge_content = gr.Textbox(
                            label="知识内容", 
                            placeholder="请输入详细的知识内容，包括概念解释、例子、注意事项等", 
                            lines=6,
                            info="内容越详细，AI学伴的回答就越准确"
                        )
                    
                    with gr.Row():
                        add_knowledge_btn = gr.Button("📚 添加知识条目", variant="primary")
                        clear_knowledge_btn = gr.Button("🗑️ 清空输入", variant="secondary")
                    
                    add_knowledge_status = gr.Textbox(label="操作状态", interactive=False, elem_classes="status-info")
                    
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h4>💡 知识录入建议</h4>
                        <div style="background: #f7fafc; padding: 15px; border-radius: 8px; font-size: 14px; color: #4a5568;">
                            <ul style="margin: 0; padding-left: 20px;">
                                <li>标题要简洁明了，准确概括知识点</li>
                                <li>分类选择要准确，便于后续检索</li>
                                <li>标签要具体，便于AI精准匹配</li>
                                <li>内容要详细完整，包含实际应用场景</li>
                            </ul>
                        </div>
                    </div>
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>🔍 知识库浏览</h3>
                        <p style="color: #4a5568;">快速查找已有知识</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        gr.Markdown("#### 🔎 搜索条件")
                        browse_query = gr.Textbox(
                            label="搜索知识", 
                            placeholder="输入关键词搜索，如：函数、语法、公式等",
                            info="支持模糊搜索，输入关键词即可"
                        )
                        
                        with gr.Row():
                            browse_btn = gr.Button("🔍 搜索知识库", variant="primary")
                            refresh_knowledge_btn = gr.Button("🔄 刷新列表", variant="secondary")
                    
                    browse_results = gr.JSON(
                        label="📋 知识库内容", 
                        height=400,
                        elem_classes="data-table"
                    )
        
        with gr.Tab("👥 AI学伴分身"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>🎭 创建AI分身</h3>
                        <p style="color: #4a5568;">创建您的专属AI学伴分身</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        gr.Markdown("#### 👤 基本信息")
                        avatar_name = gr.Textbox(
                            label="分身名称", 
                            placeholder="给你的分身起个独特的名字，如：数学小助手、英语达人等",
                            info="分身名称将用于标识和调用"
                        )
                        
                        gr.Markdown("#### 🎨 个性化设置")
                        avatar_personality = gr.Textbox(
                            label="个性描述", 
                            placeholder="描述分身的个性特点，如：幽默风趣、严谨专业、温柔耐心、活泼开朗等", 
                            lines=3,
                            info="个性化的个性描述让分身更有特色"
                        )
                        avatar_teaching_style = gr.Textbox(
                            label="教学风格", 
                            placeholder="描述教学风格，如：启发式、案例式、互动式、循序渐进等",
                            info="教学风格决定了分身的教学方式"
                        )
                        
                        gr.Markdown("#### 🎯 专业能力")
                        avatar_expertise = gr.Textbox(
                            label="专业领域", 
                            placeholder="输入专业领域，用逗号分隔，如：数学,物理,编程,英语,历史等",
                            info="专业领域决定了分身的知识范围和专长"
                        )
                        
                        gr.Markdown("#### 🔧 高级设置")
                        avatar_prompt = gr.Textbox(
                            label="自定义提示词", 
                            placeholder="额外的个性化提示词，如：使用生动的例子、结合生活实际等", 
                            lines=3,
                            info="自定义提示词可以进一步个性化分身的行为"
                        )
                    
                    with gr.Row():
                        create_avatar_btn = gr.Button("🎭 创建分身", variant="primary")
                        clear_avatar_btn = gr.Button("🗑️ 清空输入", variant="secondary")
                    
                    create_avatar_status = gr.Textbox(label="创建状态", interactive=False, elem_classes="status-info")
                    
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>📚 分身知识管理</h3>
                        <p style="color: #4a5568;">为分身添加专业知识</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        gr.Markdown("#### 📖 知识录入")
                        avatar_select_for_knowledge = gr.Dropdown(
                            label="选择分身", 
                            choices=[],
                            info="选择要添加知识的分身"
                        )
                        avatar_knowledge_topic = gr.Textbox(
                            label="知识主题", 
                            placeholder="输入知识主题，如：二次函数的性质",
                            info="简洁明了的知识主题"
                        )
                        avatar_knowledge_content = gr.Textbox(
                            label="知识内容", 
                            placeholder="输入详细的知识内容，包括概念、例子、注意事项等", 
                            lines=4,
                            info="详细的内容让分身回答更准确"
                        )
                    
                    add_avatar_knowledge_btn = gr.Button("📚 为分身添加知识", variant="secondary")
                    add_avatar_knowledge_status = gr.Textbox(label="添加状态", interactive=False, elem_classes="status-info")
                
                with gr.Column():
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>💬 我的分身</h3>
                        <p style="color: #4a5568;">管理和使用您的AI分身</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        gr.Markdown("#### 🎯 分身选择")
                        my_avatars_dropdown = gr.Dropdown(
                            label="选择分身", 
                            choices=[],
                            info="选择要使用的AI分身"
                        )
                        refresh_avatars_btn = gr.Button("🔄 刷新分身列表", variant="secondary")
                    
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>🗣️ 与分身对话</h3>
                        <p style="color: #4a5568;">与您的AI分身进行学习交流</p>
                    </div>
                    """)
                    
                    with gr.Group():
                        gr.Markdown("#### 💬 对话输入")
                        avatar_chat_input = gr.Textbox(
                            label="输入消息", 
                            placeholder="输入你想问分身的问题，如：请解释二次函数的性质...", 
                            lines=4,
                            info="详细的问题描述有助于分身给出更好的回答"
                        )
                        
                        with gr.Row():
                            avatar_chat_btn = gr.Button("📤 发送消息", variant="primary")
                            clear_chat_btn = gr.Button("🗑️ 清空对话", variant="secondary")
                    
                    avatar_chat_output = gr.Textbox(
                        label="分身回复", 
                        interactive=False, 
                        lines=6,
                        elem_classes="chat-output"
                    )
                    
                    gr.Markdown("""
                    <div class="welcome-card">
                        <h3>📊 分身服务统计</h3>
                        <p style="color: #4a5568;">查看分身的使用情况</p>
                    </div>
                    """)
                    
                    avatar_stats = gr.JSON(
                        label="📈 分身统计信息", 
                        height=250,
                        elem_classes="stats-panel"
                    )
        
        with gr.Tab("ℹ️ 系统信息"):
            gr.Markdown("""
            <div class="welcome-card">
                <h2>📋 关于HAI学伴系统</h2>
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; margin: 20px 0;">
                    <h3 style="margin: 0 0 10px 0;">🌟 系统理念</h3>
                    <p style="margin: 0; font-size: 16px; font-weight: 300;">"愿人人皆有学伴，处处皆有智慧"</p>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0;">
                <div class="feature-card">
                    <h4>📊 系统信息</h4>
                    <ul style="color: #4a5568; line-height: 1.6;">
                        <li><strong>版本：</strong>v2.0.0</li>
                        <li><strong>开发者：</strong>AI学伴团队</li>
                        <li><strong>更新日期：</strong>2025年</li>
                        <li><strong>用户界面：</strong>全新美化设计</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <h4>🎯 核心功能</h4>
                    <ul style="color: #4a5568; line-height: 1.6;">
                        <li>🤖 智能AI对话学习伙伴</li>
                        <li>👥 AI学伴分身创建与管理</li>
                        <li>📊 学习进度可视化仪表板</li>
                        <li>📚 知识库管理与搜索</li>
                        <li>🎯 个性化学习目标设定</li>
                        <li>📈 学习数据统计分析</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <h4>🔧 技术架构</h4>
                    <ul style="color: #4a5568; line-height: 1.6;">
                        <li><strong>前端：</strong>Gradio + 自定义CSS</li>
                        <li><strong>后端：</strong>Python + SQLite</li>
                        <li><strong>AI引擎：</strong>scikit-learn + OpenAI API</li>
                        <li><strong>数据可视化：</strong>Plotly + Pandas</li>
                        <li><strong>知识检索：</strong>RAG技术实现</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <h4>✨ 界面特色</h4>
                    <ul style="color: #4a5568; line-height: 1.6;">
                        <li>🎨 现代化卡片式布局</li>
                        <li>🌈 渐变色彩主题设计</li>
                        <li>📱 响应式界面适配</li>
                        <li>🎯 直观的图标和提示</li>
                        <li>⚡ 流畅的用户交互体验</li>
                    </ul>
                </div>
            </div>
            
            <div class="welcome-card">
                <h3>🚀 快速开始指南</h3>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea;">
                    <ol style="color: #4a5568; line-height: 1.8; margin: 0; padding-left: 20px;">
                        <li><strong>注册账户：</strong>点击主页的"注册新用户"按钮创建账户</li>
                        <li><strong>完善档案：</strong>在用户管理中填写详细的学习者信息</li>
                        <li><strong>设定目标：</strong>在学习管理中设置具体的学习目标</li>
                        <li><strong>开始对话：</strong>与AI学伴进行智能学习对话</li>
                        <li><strong>记录进度：</strong>定期更新学习进度和时间投入</li>
                        <li><strong>创建分身：</strong>在AI学伴分身中创建您的专属数字分身</li>
                        <li><strong>管理知识：</strong>为分身添加专业知识，提升回答质量</li>
                    </ol>
                </div>
            </div>
            
            <div class="welcome-card">
                <h3>💡 使用技巧</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                    <div style="background: #e8f4fd; padding: 15px; border-radius: 8px;">
                        <h4 style="color: #1976d2; margin: 0 0 10px 0;">🎯 目标设定</h4>
                        <p style="color: #4a5568; margin: 0; font-size: 14px;">设定具体、可衡量、可达成的学习目标，定期回顾和调整</p>
                    </div>
                    <div style="background: #f3e5f5; padding: 15px; border-radius: 8px;">
                        <h4 style="color: #7b1fa2; margin: 0 0 10px 0;">📊 进度跟踪</h4>
                        <p style="color: #4a5568; margin: 0; font-size: 14px;">及时记录学习进度，利用仪表板分析学习模式和效果</p>
                    </div>
                    <div style="background: #e8f5e8; padding: 15px; border-radius: 8px;">
                        <h4 style="color: #388e3c; margin: 0 0 10px 0;">🤖 AI对话</h4>
                        <p style="color: #4a5568; margin: 0; font-size: 14px;">与AI学伴进行深度对话，探索知识点的不同角度</p>
                    </div>
                    <div style="background: #fff3e0; padding: 15px; border-radius: 8px;">
                        <h4 style="color: #f57c00; margin: 0 0 10px 0;">👥 分身创建</h4>
                        <p style="color: #4a5568; margin: 0; font-size: 14px;">创建多个专业分身，满足不同学科和学习场景的需求</p>
                    </div>
                </div>
            </div>
            """)
        
        # 事件处理函数
        def update_learner_fields_visibility(user_role):
            return gr.update(visible=(user_role == "learner"))
        
        def on_login(username, password):
            status, user = login_user(username, password)
            if user:
                user_info = f"用户名: {user['username']}\n用户类型: {user['user_type']}\n用户角色: {user['user_role']}"
                return status, user, user_info, gr.update(value="")
            return status, None, "", gr.update(value="")
        
        def on_logout():
            global current_user
            current_user = None
            return "已成功登出", None, "", ""
        
        def on_register(username, password, user_type, user_role, full_name, age, grade_level,
                       learning_goals, preferred_subjects, learning_style, weekly_study_hours):
            # 处理学习目标字符串
            goals_list = [g.strip() for g in learning_goals.split(",") if g.strip()] if learning_goals else []
            subjects_list = [s.strip() for s in preferred_subjects.split(",") if s.strip()] if preferred_subjects else []
            
            status = register_user(username, password, user_type, user_role, full_name, 
                                 age, grade_level, goals_list, subjects_list, learning_style, 
                                 weekly_study_hours)
            return status
        
        def on_chat(message, user_state):
            if not user_state:
                return "请先登录后再与AI学伴对话。", ""
            
            response = chat_with_ai(message, user_state['user_type'])
            return response, ""
        
        def on_refresh_history(user_state):
            if not user_state:
                return []
            
            history = get_chat_history(user_state['id'])
            return history
        
        def on_add_goal(title, description, target_date, priority, user_state):
            if not user_state:
                return "请先登录后再添加学习目标。"
            
            return add_learning_goal(user_state['id'], title, description, target_date, priority)
        
        def on_update_progress(subject, topic, percentage, mastery, study_time, user_state):
            if not user_state:
                return "请先登录后再更新学习进度。"
            
            return update_learning_progress(user_state['id'], subject, topic, percentage, mastery, study_time)
        
        def on_generate_dashboard(user_state):
            if not user_state:
                return "请先登录后再查看学习仪表板。"
            
            return create_learning_dashboard(user_state['id'])
        
        def on_add_knowledge(topic, content, subject, difficulty, tags):
            return add_knowledge_to_base(topic, content, subject, difficulty, tags)
        
        def on_search_knowledge(query):
            results = search_knowledge_base(query)
            return results
        
        def on_quick_login(username, password):
            status, user = login_user(username, password)
            return status
        
        # AI分身相关函数
        def on_create_avatar(name, personality, expertise, teaching_style, prompt, user_state):
            if not user_state:
                return "请先登录后再创建分身。"
            
            expertise_list = [e.strip() for e in expertise.split(",") if e.strip()] if expertise else []
            return create_ai_avatar(user_state['id'], name, personality, expertise_list, teaching_style, prompt)
        
        def on_refresh_avatars(user_state):
            if not user_state:
                return [], "请先登录。", []
            
            avatars = get_user_avatars(user_state['id'])
            avatar_choices = [(f"{avatar[1]} (ID: {avatar[0]})", avatar[0]) for avatar in avatars]
            avatar_names = [avatar[1] for avatar in avatars]
            
            return avatar_choices, gr.update(choices=avatar_choices), gr.update(choices=avatar_choices)
        
        def on_avatar_chat(message, avatar_id, user_state):
            if not user_state:
                return "请先登录后再与分身对话。"
            
            if not avatar_id:
                return "请先选择一个分身。"
            
            # 获取学习者档案
            learner_profile = get_learner_profile(user_state['id'])
            return chat_with_avatar(avatar_id, message, learner_profile)
        
        def on_add_avatar_knowledge(avatar_id, topic, content, user_state):
            if not user_state:
                return "请先登录。"
            
            if not avatar_id:
                return "请先选择一个分身。"
            
            return add_avatar_knowledge(avatar_id, topic, content)
        
        # 绑定事件
        reg_user_role.change(update_learner_fields_visibility, inputs=[reg_user_role], outputs=[learner_fields])
        
        quick_login_btn.click(on_quick_login, inputs=[quick_username, quick_password], outputs=[quick_login_status])
        
        login_btn.click(on_login, inputs=[login_username, login_password], 
                       outputs=[login_status, user_state, current_user_info, login_password])
        
        logout_btn.click(on_logout, outputs=[logout_status, user_state, current_user_info, login_status])
        
        reg_btn.click(on_register, 
                     inputs=[reg_username, reg_password, reg_user_type, reg_user_role, 
                            reg_full_name, reg_age, reg_grade_level, reg_learning_goals,
                            reg_preferred_subjects, reg_learning_style, reg_weekly_study_hours],
                     outputs=[reg_status])
        
        chat_btn.click(on_chat, inputs=[chat_input, user_state], outputs=[chat_output, chat_input])
        
        refresh_history_btn.click(on_refresh_history, inputs=[user_state], outputs=[chat_history])
        
        add_goal_btn.click(on_add_goal, 
                          inputs=[goal_title, goal_description, goal_target_date, goal_priority, user_state],
                          outputs=[goal_status])
        
        update_progress_btn.click(on_update_progress,
                                 inputs=[progress_subject, progress_topic, progress_percentage,
                                        progress_mastery, progress_study_time, user_state],
                                 outputs=[progress_status])
        
        dashboard_btn.click(on_generate_dashboard, inputs=[user_state], outputs=[learning_dashboard])
        
        add_knowledge_btn.click(on_add_knowledge,
                               inputs=[knowledge_topic, knowledge_content, knowledge_subject,
                                      knowledge_difficulty, knowledge_tags],
                               outputs=[add_knowledge_status])
        
        browse_btn.click(on_search_knowledge, inputs=[browse_query], outputs=[browse_results])
        
        knowledge_search_btn.click(on_search_knowledge, inputs=[knowledge_query], outputs=[knowledge_results])
        
        # AI分身事件绑定
        create_avatar_btn.click(on_create_avatar,
                               inputs=[avatar_name, avatar_personality, avatar_expertise, 
                                      avatar_teaching_style, avatar_prompt, user_state],
                               outputs=[create_avatar_status])
        
        refresh_avatars_btn.click(on_refresh_avatars, inputs=[user_state], 
                                 outputs=[my_avatars_dropdown, avatar_select_for_knowledge, my_avatars_dropdown])
        
        avatar_chat_btn.click(on_avatar_chat,
                             inputs=[avatar_chat_input, my_avatars_dropdown, user_state],
                             outputs=[avatar_chat_output])
        
        add_avatar_knowledge_btn.click(on_add_avatar_knowledge,
                                       inputs=[avatar_select_for_knowledge, avatar_knowledge_topic, 
                                              avatar_knowledge_content, user_state],
                                       outputs=[add_avatar_knowledge_status])
        
        # 页面加载时刷新分身列表
        app.load(on_refresh_avatars, inputs=[user_state], 
                outputs=[my_avatars_dropdown, avatar_select_for_knowledge, my_avatars_dropdown])
    
    app._hai_theme = app_theme
    app._hai_css = custom_css
    return app

# 创建并启动应用
if __name__ == "__main__":
    # 创建Gradio界面
    app = create_gradio_interface()
    
    # 启动应用
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7861,
        "share": True,
        "inbrowser": True,
        "show_error": True,
    }
    try:
        app.launch(
            **launch_kwargs,
            theme=getattr(app, "_hai_theme", None),
            css=getattr(app, "_hai_css", None),
        )
    except TypeError:
        app.launch(**launch_kwargs)