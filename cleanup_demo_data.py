import sqlite3

def cleanup_demo_data():
    """清理演示用数据"""
    conn = sqlite3.connect('hai_learn.db')
    c = conn.cursor()
    
    try:
        # 清空知识库
        c.execute("DELETE FROM knowledge_base")
        kb_count = c.rowcount
        
        # 清空对话历史
        c.execute("DELETE FROM conversations")
        conv_count = c.rowcount
        
        # 清空功德积分记录
        c.execute("DELETE FROM merit_points")
        mp_count = c.rowcount
        
        # 清空学习目标
        c.execute("DELETE FROM learning_goals")
        lg_count = c.rowcount
        
        # 清空学习笔记
        c.execute("DELETE FROM study_notes")
        sn_count = c.rowcount
        
        # 清空学习进度
        c.execute("DELETE FROM learning_progress")
        lp_count = c.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"演示数据清理成功！")
        print(f"- 知识库记录: {kb_count} 条")
        print(f"- 对话历史: {conv_count} 条")
        print(f"- 功德积分记录: {mp_count} 条")
        print(f"- 学习目标: {lg_count} 条")
        print(f"- 学习笔记: {sn_count} 条")
        print(f"- 学习进度: {lp_count} 条")
        
    except Exception as e:
        conn.close()
        print(f"清理失败: {str(e)}")

if __name__ == "__main__":
    cleanup_demo_data()
