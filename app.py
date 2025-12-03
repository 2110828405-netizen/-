import os
import io
from flask import Flask, request, jsonify, send_from_directory
from models import db, Image
from image_match import ImageSimilarityMatcher
from crawler import ImageCrawler
from config import config
from PIL import Image as PILImage  # 新增这行

# 初始化Flask应用
app = Flask(__name__)
# 加载配置
app.config.from_object(config['default'])

# 初始化数据库
db.init_app(app)

# 在应用上下文内创建数据库表（如果不存在）
with app.app_context():
    db.create_all()

# 初始化相似度匹配器
matcher = ImageSimilarityMatcher(db=db,model_name="facebook/dinov2-base")
# 预先加载特征索引（启动服务器时加载一次）

# 关键：在 Flask 应用上下文内加载特征索引（解决数据库查询报错）
with app.app_context():
    print("开始加载特征索引...")
    matcher.load_image_features_from_db()
    print(f"特征索引加载完成，有效图片数量: {len(matcher.image_paths) if matcher.image_paths else 0}")

# 初始化爬虫（需要传入app实例）
crawler = ImageCrawler(app=app, delay=1.5)


# --- API 路由 ---

@app.route('/api/search', methods=['POST'])
def search_similar():
    if 'image' not in request.files:
        return jsonify({"error": "未上传图片"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "未选择图片"}), 400

    top_k = request.form.get('top_k', 1, type=int)

    try:
        # 直接从内存读取图片
        image_bytes = file.read()
        image = PILImage.open(io.BytesIO(image_bytes))

        # 添加调试信息
        print(f"开始搜索相似图片，数据库图片数量: {len(matcher.image_paths) if matcher.image_paths else 0}")
        print(f"特征索引状态: {'已构建' if matcher.index else '未构建'}")

        # 执行相似度搜索
        results = matcher.find_most_similar(image, top_k=top_k)

        #确保所有结果都可以JSON序列化
        serializable_results = []
        for result in results:
            serializable_results.append({
                'image_id': int(result['image_id']),
                'path': str(result['path']),
                'similarity_percent': float(result['similarity_percent'])
            })

        # 添加结果调试
        print(f"搜索结果: {serializable_results}")

        return jsonify({"results": serializable_results})

    except Exception as e:
        print(f"搜索过程错误: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/images/<int:image_id>')
def get_image(image_id):
    """
    通过ID获取图片文件
    """
    try:
        image = Image.query.get_or_404(image_id)

        # 添加调试信息
        print(f"请求图片ID: {image_id}")
        print(f"图片路径: {image.path}")
        print(f"工作目录: {os.getcwd()}")
        print(f"完整路径: {os.path.join(os.getcwd(), image.path)}")
        print(f"文件存在: {os.path.exists(os.path.join(os.getcwd(), image.path))}")

        # 检查文件是否存在
        full_path = os.path.join(os.getcwd(), image.path)
        if not os.path.exists(full_path):
            print(f"错误：图片文件不存在: {full_path}")
            return jsonify({"error": "图片文件不存在", "path": image.path}), 404

        # 确保文件是有效的图片
        try:
            with PILImage.open(full_path) as img:
                # 验证是有效图片
                img.verify()
        except Exception as e:
            print(f"错误：图片文件损坏: {str(e)}")
            return jsonify({"error": "图片文件损坏", "path": image.path}), 400

        # 发送文件 - 使用 send_file 替代 send_from_directory
        from flask import send_file
        return send_file(full_path)

    except Exception as e:
        print(f"获取图片失败: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/crawl', methods=['POST'])
def trigger_crawl():
    """
    触发Pexels爬虫任务（仅支持Pexels，已删除其他来源）
    请求体（JSON）：
        {
            "category": "自然",  // 必选，中文类别（如"自然""建筑""人物"）
            "count": 10,         // 可选，爬取数量（默认10张，最大80张）
            "page": 1            // 可选，爬取页码（默认第1页）
        }
    返回：JSON格式的爬取结果
    """
    # 解析请求体中的JSON数据
    request_data = request.get_json()
    if not request_data:
        return jsonify({"error": "请求体必须为JSON格式"}), 400

    # 1. 获取必要参数（category为必选，count和page为可选）
    target_category = request_data.get('category')
    crawl_count = request_data.get('count', 10)  # 默认爬10张
    crawl_page = request_data.get('page', 1)     # 默认爬第1页

    # 2. 校验必选参数
    if not target_category:
        return jsonify({
            "error": "缺少必选参数「category」",
            "tips": f"支持的类别：{list(crawler.category_mappings.keys())}"
        }), 400

    # 3. 调用Pexels爬虫（确保在Flask应用上下文中执行数据库操作）
    try:
        with app.app_context():
            # 直接调用crawl_pexels，传递类别、数量、页码
            success_count = crawler.crawl_pexels(
                category=target_category,
                count=crawl_count,
                page=crawl_page
            )

        # 4. 爬虫完成后，重建相似度匹配器的特征索引（确保新爬的图片能被搜索到）
        matcher.load_image_features_from_db(rebuild_index=True)

        # 5. 返回成功结果
        return jsonify({
            "status": "success",
            "message": f"Pexels爬虫任务完成",
            "category": target_category,
            "requested_count": crawl_count,
            "success_count": success_count,  # 实际成功下载的数量
            "page": crawl_page,
            "save_dir": crawler.save_base_dir  # 图片保存的根目录
        })

    except Exception as e:
        # 捕获异常并返回错误信息
        return jsonify({
            "status": "error",
            "error": f"爬虫任务失败：{str(e)}"
        }), 500


@app.route('/api/images', methods=['GET'])
def list_images():
    """
    列出数据库中的所有图片（简化版）
    """
    images = Image.query.all()
    result = [{
        "id": img.id,
        "path": img.path,
        "category": img.category,
        "source": img.source
    } for img in images]
    return jsonify({"images": result})

@app.route('/')
def index():
    """根路径默认页面"""
    # 检查是否存在static/index.html文件，如果存在则返回页面，否则返回API信息
    if os.path.exists(os.path.join('static', 'index.html')):
        return send_from_directory('static', 'index.html')
    else:
        return jsonify({
            "message": "欢迎使用图片相似度搜索API",
            "available_endpoints": {
                "/api/search (POST)": "上传图片搜索相似图片",
                "/api/images/<image_id> (GET)": "通过ID获取图片",
                "/api/crawl (POST)": "触发图片爬取任务",
                "/api/images (GET)": "列出所有图片"
            }
        })

# --- 启动服务器 ---
# --- 启动服务器 ---
if __name__ == '__main__':
    # 新增：启动服务器时自动批量爬取大量图片（数据库为空才爬）
    with app.app_context():
        image_count = Image.query.count()
        if image_count == 0:
            print("\n数据库为空，开始自动批量爬取图片（约100张）...")
            # 定义要爬取的类别和每个类别的数量（可按需修改）
            crawl_config = [
                ("自然", 20),
                ("建筑", 20),
                ("人物", 20),
                ("动物", 20),
                ("食物", 20)
            ]
            # 循环爬取每个类别
            total_crawled = 0
            for category, count in crawl_config:
                crawled = crawler.crawl_pexels(category=category, count=count)
                total_crawled += crawled
                print(f"✅ {category} 类爬取完成，新增 {crawled} 张")

            # 爬完所有类别后，重建特征索引
            print(f"\n📊 批量爬取完成，共新增 {total_crawled} 张图片，开始构建特征索引...")
            matcher.load_image_features_from_db(rebuild_index=True)
            print("🔍 特征索引构建完成，用户可直接在网页比对！")
        else:
            print(f"\n📦 数据库已有 {image_count} 张图片，跳过自动爬取")

    # 原有启动服务器的代码（不变）
    app.run(host='0.0.0.0', port=5000, debug=False)