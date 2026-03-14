from flask import Flask, render_template, request, jsonify, session, redirect, send_from_directory
import sqlite3
import hashlib
import secrets
from datetime import datetime
import json
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# 文件上传配置
UPLOAD_FOLDER = 'uploads/avatars'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 数据库初始化
def init_db():
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    # 用户表 - 扩展学习者账户功能
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  user_type TEXT DEFAULT 'free',  -- free, paid, learner
                  user_role TEXT DEFAULT 'user',   -- user, learner, tutor
                  credits INTEGER DEFAULT 0,
                  avatar_url TEXT,
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
                  total_study_time INTEGER DEFAULT 0,  -- 分钟
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
                  mastery_level TEXT DEFAULT 'beginner',  -- beginner, intermediate, advanced
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
                  priority TEXT DEFAULT 'medium',  -- low, medium, high
                  status TEXT DEFAULT 'active',   -- active, completed, paused
                  progress_notes TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # 学习笔记表
    c.execute('''CREATE TABLE IF NOT EXISTS study_notes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  title TEXT,
                  content TEXT,
                  subject TEXT,
                  tags TEXT,
                  is_shared BOOLEAN DEFAULT FALSE,
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
    
    # AI API配置表
    c.execute('''CREATE TABLE IF NOT EXISTS api_configs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER UNIQUE,
                  api_provider TEXT DEFAULT 'openai',
                  api_key TEXT,
                  base_url TEXT,
                  model_name TEXT DEFAULT 'gpt-3.5-turbo',
                  is_active BOOLEAN DEFAULT FALSE,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
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
    
    def generate_response(self, user_message, user_type='free', api_config=None):
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
        
        # 如果用户配置了API，使用用户的API
        if api_config and api_config.get('is_active') and api_config.get('api_key'):
            try:
                return self.call_user_api(user_message, system_prompt, api_config)
            except Exception as e:
                return f"API调用失败: {str(e)}。使用备用回复：{self.simulate_ai_response(user_message, system_prompt)}"
        
        # 使用默认模拟回复
        return self.simulate_ai_response(user_message, system_prompt)
    
    def call_user_api(self, user_message, system_prompt, api_config):
        """调用用户配置的API"""
        from openai import OpenAI
        
        api_key = api_config.get('api_key')
        base_url = api_config.get('base_url')
        model_name = api_config.get('model_name', 'gpt-3.5-turbo')
        
        client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def simulate_ai_response(self, user_message, system_prompt):
        """模拟AI回复（演示用）"""
        # 基础回复模板
        responses = [
            f"这是一个很好的问题！关于'{user_message}'，让我来帮你解答。",
            f"我理解你的疑问。'{user_message}'这个问题很有深度，我来详细解释一下。",
            f"很高兴能帮助你学习！关于'{user_message}'，这里是我的建议。"
        ]
        
        # 根据知识库内容增强回复
        if "知识库" in system_prompt:
            return f"基于我的知识库，我找到了相关的信息来帮助你理解'{user_message}'。让我给你一个全面的解答。"
        
        return responses[hash(user_message) % len(responses)]

# 初始化AI学伴
ai_companion = HAILearningCompanion()

@app.route('/')
def index():
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('chat.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    c.execute("SELECT id, password_hash, user_type FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[1], password):
        session['user_id'] = user[0]
        session['username'] = username
        session['user_type'] = user[2]
        return jsonify({'success': True, 'user_type': user[2]})
    
    return jsonify({'success': False, 'message': '用户名或密码错误'})

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    user_type = data.get('user_type', 'free')
    user_role = data.get('user_role', 'user')
    learner_data = data.get('learner_data')
    
    if not username or not password:
        return jsonify({'success': False, 'message': '用户名和密码不能为空'})
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        password_hash = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password_hash, user_type, user_role) VALUES (?, ?, ?, ?)", 
                  (username, password_hash, user_type, user_role))
        
        user_id = c.lastrowid
        
        # 如果是学习者角色，创建学习者档案
        if user_role == 'learner' and learner_data:
            c.execute("""
                INSERT INTO learner_profiles 
                (user_id, full_name, age, grade_level, learning_goals, 
                 preferred_subjects, learning_style, weekly_study_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                learner_data.get('full_name', username),
                learner_data.get('age'),
                learner_data.get('grade_level'),
                json.dumps(learner_data.get('learning_goals', [])),
                json.dumps(learner_data.get('preferred_subjects', [])),
                learner_data.get('learning_style'),
                learner_data.get('weekly_study_hours', 0)
            ))
        
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'success': False, 'message': '用户名已存在'})
    except Exception as e:
        conn.close()
        return jsonify({'success': False, 'message': f'注册失败: {str(e)}'})

@app.route('/chat')
def chat_page():
    """显示聊天页面"""
    if 'user_id' not in session:
        return redirect('/')
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """处理聊天消息"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.json
    message = data.get('message')
    
    # 获取用户的API配置
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    c.execute("SELECT api_provider, api_key, base_url, model_name, is_active FROM api_configs WHERE user_id = ?", 
              (session['user_id'],))
    api_config_data = c.fetchone()
    conn.close()
    
    api_config = None
    if api_config_data:
        api_config = {
            'api_provider': api_config_data[0],
            'api_key': api_config_data[1],
            'base_url': api_config_data[2],
            'model_name': api_config_data[3],
            'is_active': api_config_data[4]
        }
    
    # 生成AI回复
    response = ai_companion.generate_response(message, session.get('user_type', 'free'), api_config)
    
    # 保存对话记录
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    c.execute("INSERT INTO conversations (user_id, message, response) VALUES (?, ?, ?)",
              (session['user_id'], message, response))
    
    # 免费用户获得功德积分
    if session.get('user_type') == 'free':
        c.execute("UPDATE users SET credits = credits + 1 WHERE id = ?", (session['user_id'],))
        c.execute("INSERT INTO merit_points (user_id, points, reason) VALUES (?, ?, ?)",
                  (session['user_id'], 1, "免费帮助学习者"))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'response': response})

@app.route('/history')
def history():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    c.execute("SELECT message, response, timestamp FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20",
              (session['user_id'],))
    conversations = c.fetchall()
    conn.close()
    
    return jsonify({'success': True, 'history': conversations})

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    # 获取用户基本信息
    c.execute("SELECT id, username, user_type, user_role, created_at FROM users WHERE id = ?", (session['user_id'],))
    user = c.fetchone()
    
    if not user:
        conn.close()
        return jsonify({'success': False, 'message': '用户不存在'})
    
    # 获取学习者档案信息
    learner_profile = None
    if user[3] == 'learner':
        c.execute("""
            SELECT full_name, age, grade_level, learning_goals, 
                   preferred_subjects, learning_style, weekly_study_hours
            FROM learner_profiles WHERE user_id = ?
        """, (session['user_id'],))
        learner_data = c.fetchone()
        if learner_data:
            learner_profile = {
                'full_name': learner_data[0],
                'age': learner_data[1],
                'grade_level': learner_data[2],
                'learning_goals': json.loads(learner_data[3]) if learner_data[3] else [],
                'preferred_subjects': json.loads(learner_data[4]) if learner_data[4] else [],
                'learning_style': learner_data[5],
                'weekly_study_hours': learner_data[6]
            }
    
    # 获取学习进度
    c.execute("""
        SELECT subject, topic, progress_percentage, mastery_level, 
               study_time_minutes, last_studied
        FROM learning_progress WHERE user_id = ?
        ORDER BY last_studied DESC
    """, (session['user_id'],))
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
        SELECT id, goal_title, goal_description, target_date, priority, status, progress_notes
        FROM learning_goals WHERE user_id = ?
        ORDER BY target_date ASC
    """, (session['user_id'],))
    learning_goals = []
    for row in c.fetchall():
        learning_goals.append({
            'id': row[0],
            'goal_title': row[1],
            'goal_description': row[2],
            'target_date': row[3],
            'priority': row[4],
            'status': row[5],
            'progress_notes': row[6]
        })
    
    conn.close()
    
    return jsonify({
        'success': True,
        'user': {
            'id': user[0],
            'username': user[1],
            'user_type': user[2],
            'user_role': user[3],
            'created_at': user[4]
        },
        'learner_profile': learner_profile,
        'learning_progress': learning_progress,
        'learning_goals': learning_goals
    })

@app.route('/learner/dashboard')
def learner_dashboard():
    """学习者仪表板页面"""
    if 'user_id' not in session:
        return redirect('/login')
    
    return render_template('learner_dashboard.html')

# 学习者账户管理功能
@app.route('/learner/profile/update', methods=['POST'])
def update_learner_profile():
    """更新学习者档案"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.json
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        # 检查是否已存在学习者档案
        c.execute("SELECT id FROM learner_profiles WHERE user_id = ?", (session['user_id'],))
        existing = c.fetchone()
        
        if existing:
            # 更新现有档案
            c.execute("""
                UPDATE learner_profiles 
                SET full_name = ?, age = ?, grade_level = ?, learning_goals = ?, 
                    preferred_subjects = ?, learning_style = ?, weekly_study_hours = ?,
                    updated_at = ?
                WHERE user_id = ?
            """, (
                data.get('full_name'),
                data.get('age'),
                data.get('grade_level'),
                json.dumps(data.get('learning_goals', [])),
                json.dumps(data.get('preferred_subjects', [])),
                data.get('learning_style'),
                data.get('weekly_study_hours', 0),
                datetime.now(),
                session['user_id']
            ))
        else:
            # 创建新档案
            c.execute("""
                INSERT INTO learner_profiles 
                (user_id, full_name, age, grade_level, learning_goals, 
                 preferred_subjects, learning_style, weekly_study_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session['user_id'],
                data.get('full_name'),
                data.get('age'),
                data.get('grade_level'),
                json.dumps(data.get('learning_goals', [])),
                json.dumps(data.get('preferred_subjects', [])),
                data.get('learning_style'),
                data.get('weekly_study_hours', 0)
            ))
        
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '学习者档案更新成功'})
    except Exception as e:
        conn.close()
        return jsonify({'success': False, 'message': f'更新失败: {str(e)}'})

@app.route('/learner/goals/add', methods=['POST'])
def add_learning_goal():
    """添加学习目标"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.json
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        c.execute("""
            INSERT INTO learning_goals 
            (user_id, goal_title, goal_description, target_date, priority, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session['user_id'],
            data.get('goal_title'),
            data.get('goal_description'),
            data.get('target_date'),
            data.get('priority', 'medium'),
            'active'
        ))
        
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '学习目标添加成功'})
    except Exception as e:
        conn.close()
        return jsonify({'success': False, 'message': f'添加失败: {str(e)}'})

@app.route('/learner/goals/update/<int:goal_id>', methods=['POST'])
def update_learning_goal(goal_id):
    """更新学习目标状态"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.json
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        c.execute("""
            UPDATE learning_goals 
            SET status = ?, progress_notes = ?
            WHERE id = ? AND user_id = ?
        """, (
            data.get('status'),
            data.get('progress_notes'),
            goal_id,
            session['user_id']
        ))
        
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '学习目标更新成功'})
    except Exception as e:
        conn.close()
        return jsonify({'success': False, 'message': f'更新失败: {str(e)}'})

@app.route('/learner/notes/add', methods=['POST'])
def add_study_note():
    """添加学习笔记"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.json
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        c.execute("""
            INSERT INTO study_notes 
            (user_id, title, content, subject, tags, is_shared)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session['user_id'],
            data.get('title'),
            data.get('content'),
            data.get('subject'),
            json.dumps(data.get('tags', [])),
            data.get('is_shared', False)
        ))
        
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '学习笔记添加成功'})
    except Exception as e:
        conn.close()
        return jsonify({'success': False, 'message': f'添加失败: {str(e)}'})

@app.route('/learner/notes/list')
def list_study_notes():
    """获取学习笔记列表"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    c.execute("""
        SELECT id, title, subject, tags, is_shared, created_at 
        FROM study_notes 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    """, (session['user_id'],))
    
    notes = c.fetchall()
    conn.close()
    
    notes_list = []
    for note in notes:
        notes_list.append({
            'id': note[0],
            'title': note[1],
            'subject': note[2],
            'tags': json.loads(note[3]) if note[3] else [],
            'is_shared': note[4],
            'created_at': note[5]
        })
    
    return jsonify({'success': True, 'notes': notes_list})

@app.route('/learner/progress/update', methods=['POST'])
def update_learning_progress():
    """更新学习进度"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.json
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        subject = data.get('subject')
        topic = data.get('topic')
        progress_percentage = data.get('progress_percentage', 0)
        mastery_level = data.get('mastery_level', 'beginner')
        study_time_minutes = data.get('study_time_minutes', 0)
        
        # 检查是否已存在该学习进度记录
        c.execute("""
            SELECT id FROM learning_progress 
            WHERE user_id = ? AND subject = ? AND topic = ?
        """, (session['user_id'], subject, topic))
        
        existing = c.fetchone()
        
        if existing:
            # 更新现有进度
            c.execute("""
                UPDATE learning_progress 
                SET progress_percentage = ?, mastery_level = ?, 
                    study_time_minutes = study_time_minutes + ?, last_studied = ?
                WHERE user_id = ? AND subject = ? AND topic = ?
            """, (
                progress_percentage, mastery_level, study_time_minutes,
                datetime.now(), session['user_id'], subject, topic
            ))
        else:
            # 创建新进度记录
            c.execute("""
                INSERT INTO learning_progress 
                (user_id, subject, topic, progress_percentage, mastery_level, 
                 study_time_minutes, last_studied)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session['user_id'], subject, topic, progress_percentage,
                mastery_level, study_time_minutes, datetime.now()
            ))
        
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': '学习进度更新成功'})
    except Exception as e:
        conn.close()
        return jsonify({'success': False, 'message': f'更新失败: {str(e)}'})

# AI API配置管理
@app.route('/api/config/get')
def get_api_config():
    """获取用户的API配置"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    c.execute("""
        SELECT api_provider, api_key, base_url, model_name, is_active 
        FROM api_configs WHERE user_id = ?
    """, (session['user_id'],))
    config = c.fetchone()
    conn.close()
    
    if config:
        return jsonify({
            'success': True,
            'config': {
                'api_provider': config[0],
                'api_key': '***' if config[1] else '',
                'base_url': config[2],
                'model_name': config[3],
                'is_active': bool(config[4])
            }
        })
    else:
        return jsonify({
            'success': True,
            'config': {
                'api_provider': 'openai',
                'api_key': '',
                'base_url': '',
                'model_name': 'gpt-3.5-turbo',
                'is_active': False
            }
        })

@app.route('/api/config/save', methods=['POST'])
def save_api_config():
    """保存用户的API配置"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.json
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        api_provider = data.get('api_provider', 'openai')
        api_key = data.get('api_key', '')
        base_url = data.get('base_url', '')
        model_name = data.get('model_name', 'gpt-3.5-turbo')
        is_active = data.get('is_active', False)
        
        # 检查是否已存在配置
        c.execute("SELECT id FROM api_configs WHERE user_id = ?", (session['user_id'],))
        existing = c.fetchone()
        
        if existing:
            # 更新现有配置
            if api_key:
                c.execute("""
                    UPDATE api_configs 
                    SET api_provider = ?, api_key = ?, base_url = ?, model_name = ?, 
                        is_active = ?, updated_at = ?
                    WHERE user_id = ?
                """, (api_provider, api_key, base_url, model_name, is_active, datetime.now(), session['user_id']))
            else:
                # 如果API key为空，不更新api_key字段
                c.execute("""
                    UPDATE api_configs 
                    SET api_provider = ?, base_url = ?, model_name = ?, 
                        is_active = ?, updated_at = ?
                    WHERE user_id = ?
                """, (api_provider, base_url, model_name, is_active, datetime.now(), session['user_id']))
        else:
            # 创建新配置
            c.execute("""
                INSERT INTO api_configs 
                (user_id, api_provider, api_key, base_url, model_name, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session['user_id'], api_provider, api_key, base_url, model_name, is_active))
        
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'API配置保存成功'})
    except Exception as e:
        conn.close()
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'})

@app.route('/api/config/test', methods=['POST'])
def test_api_config():
    """测试API配置是否有效"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.json
    api_key = data.get('api_key', '')
    base_url = data.get('base_url', '')
    model_name = data.get('model_name', 'gpt-3.5-turbo')
    
    if not api_key:
        return jsonify({'success': False, 'message': '请提供API Key'})
    
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None
        )
        
        # 发送一个简单的测试消息
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=10
        )
        
        return jsonify({'success': True, 'message': 'API连接测试成功！'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'API连接失败: {str(e)}'})

@app.route('/logout')
def logout():
    session.clear()
    return jsonify({'success': True})

# 头像上传和获取
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload/avatar', methods=['POST'])
def upload_avatar():
    """上传用户头像"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    if 'avatar' not in request.files:
        return jsonify({'success': False, 'message': '没有找到文件'})
    
    file = request.files['avatar']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': '文件名不能为空'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(f"{session['user_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 更新数据库
            conn = sqlite3.connect('hai_learn.db')
            c = conn.cursor()
            c.execute("UPDATE users SET avatar_url = ? WHERE id = ?", (filename, session['user_id']))
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'filename': filename, 'message': '头像上传成功'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'上传失败: {str(e)}'})
    else:
        return jsonify({'success': False, 'message': '不支持的文件格式'})

@app.route('/uploads/avatars/<filename>')
def get_avatar(filename):
    """获取用户头像"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/avatar/get')
def get_user_avatar():
    """获取当前用户的头像URL"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    c.execute("SELECT avatar_url FROM users WHERE id = ?", (session['user_id'],))
    result = c.fetchone()
    conn.close()
    
    if result and result[0]:
        return jsonify({'success': True, 'avatar_url': f'/uploads/avatars/{result[0]}', 'has_avatar': True})
    else:
        return jsonify({'success': True, 'avatar_url': '', 'has_avatar': False})

if __name__ == '__main__':
    init_db()
    # 加载示例知识库
    ai_companion.load_knowledge_base()
    app.run(debug=True, host='0.0.0.0', port=5000)