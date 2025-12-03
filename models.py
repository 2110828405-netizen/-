from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Image(db.Model):
    """图片元数据模型"""
    id = db.Column(db.Integer, primary_key=True)
    # 图片在本地的存储路径
    path = db.Column(db.String(255), unique=True, nullable=False)
    # 图片类别
    category = db.Column(db.String(50), nullable=False)
    # 图片分辨率
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    # 图片来源（如 'pexels', 'unsplash'）
    source = db.Column(db.String(20))
    # 来源网站上的图片ID
    source_id = db.Column(db.String(50))
    # 摄影师信息
    photographer = db.Column(db.String(100))
    photographer_url = db.Column(db.String(255))
    # 上传到我们系统的时间
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    # 原始图片URL
    original_url = db.Column(db.String(255))

    def __repr__(self):
        return f'<Image {self.id} - {self.category}>'