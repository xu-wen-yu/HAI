#!/usr/bin/env python3
"""
HAI学伴系统 - Gradio版本启动脚本
启动AI学习伙伴平台的现代化Web界面
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误：需要Python 3.8或更高版本")
        print(f"当前版本：{sys.version}")
        sys.exit(1)
    print(f"✅ Python版本检查通过：{sys.version.split()[0]}")

def install_requirements():
    """安装依赖包"""
    print("📦 正在检查依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖包安装完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败：{e}")
        sys.exit(1)

def init_database():
    """初始化数据库"""
    print("🗄️ 正在初始化数据库...")
    try:
        from gradio_app import init_db
        init_db()
        print("✅ 数据库初始化完成")
    except Exception as e:
        print(f"❌ 数据库初始化失败：{e}")
        sys.exit(1)

def start_application():
    """启动应用"""
    print("🚀 正在启动HAI学伴系统...")
    try:
        # 导入并启动Gradio应用
        from gradio_app import create_gradio_interface
        app = create_gradio_interface()
        
        print("🌐 应用启动成功！")
        print("📱 请访问：http://localhost:7860")
        print("⏹️  按Ctrl+C停止服务")
        
        # 启动应用
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            inbrowser=True,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n🛑 应用已停止")
    except Exception as e:
        print(f"❌ 应用启动失败：{e}")
        sys.exit(1)

def main():
    """主函数"""
    print("🎯 HAI学伴系统 - Gradio版本")
    print("=" * 50)
    
    # 检查Python版本
    check_python_version()
    
    # 检查并安装依赖
    install_requirements()
    
    # 初始化数据库
    init_database()
    
    # 启动应用
    start_application()

if __name__ == "__main__":
    main()