import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

class Config:
    """基本配置"""
    # SQLite数据库文件路径
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///image_library.db')
    # 禁用SQLAlchemy的修改跟踪功能，以提高性能
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Flask应用密钥，用于会话管理等（在生产环境中应设置为一个复杂的随机字符串）
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    # 图片存储的根目录
    IMAGE_STORAGE_BASE_DIR = os.path.join('static', 'images')
    # 确保图片存储目录存在
    os.makedirs(IMAGE_STORAGE_BASE_DIR, exist_ok=True)

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    # 在生产环境中，应从环境变量获取密钥和数据库URL
    SECRET_KEY = os.getenv('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')

# 根据环境变量选择配置
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}