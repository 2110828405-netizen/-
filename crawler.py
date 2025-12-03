import os
import time
import requests
from datetime import datetime
from tqdm import tqdm
import shutil
from models import db, Image  # 导入数据库模型


class ImageCrawler:
    def __init__(self, app, delay=1.0):
        """
        图片爬虫初始化
        Args:
            app (Flask): Flask应用实例
            delay (float): 请求延迟时间(秒)
        """
        self.app = app
        self.delay = delay

        # 从应用配置中获取图片保存目录
        self.save_base_dir = app.config['IMAGE_STORAGE_BASE_DIR']

        # 初始化Pexels API密钥（仅保留Pexels，删除Unsplash相关）
        self.pexels_api_key = os.getenv('PEXELS_API_KEY')

        # 支持的类别映射（仅保留与Pexels匹配的关键词）
        self.category_mappings = {
            "自然": ["nature", "landscape"],
            "建筑": ["architecture", "building"],
            "人物": ["people", "portrait"],
            "动物": ["animal", "pet"],
            "食物": ["food", "cooking"]
        }

    def _create_category_dir(self, category):
        """创建类别目录（工具方法）"""
        category_dir = os.path.join(self.save_base_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        return category_dir

    def _download_image(self, url, save_path):
        """下载图片（工具方法）"""
        try:
            response = requests.get(url, stream=True, timeout=15)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                return True
            return False
        except Exception as e:
            print(f"下载图片失败: {url[:50]}... Error: {str(e)}")
            return False

    def _save_to_db(self, image_info):
        """将图片信息保存到数据库（工具方法）"""
        # 检查图片是否已存在（避免重复）
        existing_image = Image.query.filter_by(path=image_info['path']).first()
        if existing_image:
            print(f"图片已存在于数据库: {image_info['path']}")
            return False

        new_image = Image(
            path=image_info['path'],
            category=image_info['category'],
            width=image_info.get('width'),
            height=image_info.get('height'),
            source='pexels',  # 固定来源为Pexels
            source_id=image_info.get('source_id'),
            photographer=image_info.get('photographer'),
            photographer_url=image_info.get('photographer_url'),
            original_url=image_info.get('original_url')
        )

        try:
            db.session.add(new_image)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            print(f"保存到数据库失败: {str(e)}")
            return False

    def crawl_pexels(self, category, count=10, page=1):
        """
        核心方法：从Pexels爬取指定类别图片并存入数据库
        Args:
            category (str): 图片类别(中文，如"自然""建筑")
            count (int): 爬取数量（默认10张）
            page (int): 爬取页码（默认第1页）
        Returns:
            int: 实际下载成功的图片数量
        """
        # 校验API密钥
        if not self.pexels_api_key:
            print("错误：未配置Pexels API密钥（请在.env文件中设置PEXELS_API_KEY）")
            return 0

        # 校验类别是否支持
        if category not in self.category_mappings:
            print(f"不支持的类别: {category}，可选类别：{list(self.category_mappings.keys())}")
            return 0

        # 创建类别目录（如static/images/自然）
        category_dir = self._create_category_dir(category)

        # 转换为Pexels支持的英文关键词（取第一个匹配关键词）
        keywords = self.category_mappings[category]
        search_keyword = keywords[0]

        # Pexels API请求配置（参考官方API文档）
        api_url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": self.pexels_api_key}
        params = {
            "query": search_keyword,
            "per_page": min(count, 80),  # Pexels限制每页最大80张
            "page": page,
            "orientation": "landscape"  # 优先爬取横屏图片（可选）
        }

        try:
            print(f"开始从Pexels爬取「{category}」类别图片（关键词：{search_keyword}）...")
            # 发送API请求
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()  # 若API返回错误（如401/429），直接抛出异常
            api_data = response.json()

            downloaded_count = 0
            # 遍历API返回的图片列表
            for photo in tqdm(api_data.get('photos', []), desc=f"下载「{category}」图片"):
                # 1. 提取图片元数据（从Pexels API响应中获取）
                image_meta = {
                    'width': photo.get('width'),
                    'height': photo.get('height'),
                    'source_id': str(photo.get('id')),  # Pexels图片唯一ID（转为字符串避免类型问题）
                    'photographer': photo.get('photographer'),
                    'photographer_url': photo.get('photographer_url'),
                    'original_url': photo.get('src', {}).get('original')  # 最高质量图片URL
                }

                # 跳过无有效URL的图片
                if not image_meta['original_url']:
                    print(f"跳过无URL的图片（ID: {image_meta['source_id']}）")
                    continue

                # 2. 生成本地保存路径（格式：static/images/类别/pexels_图片ID.扩展名）
                # 从URL中提取文件扩展名（如jpg/png）
                file_ext = image_meta['original_url'].split('.')[-1].split('?')[0]
                # 生成唯一文件名（避免重复）
                file_name = f"pexels_{image_meta['source_id']}.{file_ext}"
                save_path = os.path.join(category_dir, file_name)
                # 将本地路径补充到元数据中
                image_meta['path'] = save_path
                image_meta['category'] = category

                # 3. 下载图片并保存到数据库
                if self._download_image(image_meta['original_url'], save_path):
                    self._save_to_db(image_meta)
                    downloaded_count += 1

                # 4. 遵守爬取延迟（避免触发Pexels API速率限制）
                time.sleep(self.delay)

                # 5. 达到目标爬取数量后停止
                if downloaded_count >= count:
                    break

            print(f"\nPexels「{category}」图片爬取完成：共成功下载 {downloaded_count} 张")
            return downloaded_count

        except requests.exceptions.HTTPError as http_err:
            # 处理API常见错误（如密钥无效、请求超限）
            if response.status_code == 401:
                print("HTTP错误 401：Pexels API密钥无效或过期，请检查.env文件")
            elif response.status_code == 429:
                print("HTTP错误 429：Pexels API请求次数超限，请稍后再试")
            else:
                print(f"Pexels API请求错误（{response.status_code}）: {str(http_err)}")
            return 0
        except Exception as e:
            print(f"Pexels爬取过程异常: {str(e)}")
            return 0