import os
import numpy as np
import faiss
from PIL import Image as PILImage
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import torch
from models import Image  # 导入Image模型



class ImageSimilarityMatcher:
    """
    图像相似度匹配器（数据库版本）
    """

    def __init__(self, db, model_name="facebook/dinov2-base"):
        """
        初始化图像相似度匹配器（从数据库加载图片）

        Args:
            db (SQLAlchemy object): Flask-SQLAlchemy 的 db 实例，用于数据库操作。
            model_name (str): 预训练模型名称。
        """
        # 1. **核心改动**：接收数据库实例，不再需要文件路径
        self.db = db
        self.model_name = model_name

        # 2. 模型和处理器的初始化逻辑保持不变
        print(f"加载模型: {model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # 设置为评估模式

        # 检查是否有可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"使用设备: {self.device}")

        # 3. 初始化特征、索引和图片ID列表
        #    - self.features: 存储所有图片的特征向量
        #    - self.index: FAISS 索引
        #    - self.image_ids: 存储与特征向量对应的数据库图片ID
        self.features = None
        self.index = None
        self.image_ids = []
        self.image_paths = []

        # 4. **核心改动**：从数据库加载图片并构建特征索引
        #self.load_image_features_from_db()

    def preprocess_image(self, image):
        """
        预处理单张图片

        Args:
            image (PIL.Image or str): 图片对象或图片路径

        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        if isinstance(image, str):
            try:
                image = PILImage.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"无法打开图片文件: {str(e)}")

        # 使用处理器预处理图像
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            size=518
        )

        # 将张量移动到指定设备
        return {k: v.to(self.device) for k, v in inputs.items()}

    # 修改 extract_features 方法，增加更详细的错误处理
    def extract_features(self, image):
        """提取单张图片的特征向量"""
        try:
            inputs = self.preprocess_image(image)

            # 添加输入验证
            if inputs is None:
                raise ValueError("图片预处理失败")

            # 提取特征（不计算梯度）
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 使用pooler_output作为全局特征
            features = outputs.pooler_output.cpu().numpy()

            # 验证特征是否有效
            if features is None or np.isnan(features).any():
                raise ValueError("提取的特征包含无效值")

            # 检查特征是否全零
            if np.all(features == 0):
                print("警告: 提取的特征全为零")
                # 尝试使用其他输出
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # 归一化特征向量
            norm = np.linalg.norm(features, axis=1, keepdims=True)

            # 防止除以零
            if norm[0][0] < 1e-10:  # 如果范数接近零
                print("警告: 特征范数接近零，跳过归一化")
            else:
                features = features / norm

            # 再次验证归一化后的特征
            if np.isnan(features).any() or np.isinf(features).any():
                raise ValueError("特征归一化失败")

            # 检查归一化后的范数
            norm_after = np.linalg.norm(features, axis=1, keepdims=True)

            return features

        except Exception as e:
            print(f"特征提取详细错误: {str(e)}")
            raise RuntimeError(f"特征提取失败: {str(e)}")
    def load_image_features_from_db(self, rebuild_index=False):
        """
        从数据库加载所有图片路径并提取特征，构建或更新FAISS索引。
        """
        # 如果索引已存在且不强制重建，则直接返回
        if self.index is not None and not rebuild_index:
            print("特征索引已存在，无需重新加载。")
            return

        print("从数据库加载图片信息...")

        # --- 核心改动：从数据库查询图片路径 ---
        # 使用 SQLAlchemy 的查询接口获取所有图片记录
        # 为了提高效率，可以只查询需要的字段：id 和 path
        db_images = Image.query.with_entities(Image.id, Image.path).all()

        if not db_images:
            print("数据库中没有图片记录。")
            self._reset_index()
            return

        # 分离ID和路径，并存入类的属性中
        self.image_ids = [img.id for img in db_images]
        self.image_paths = [img.path for img in db_images]

        print(f"找到 {len(self.image_paths)} 张图片，开始提取特征...")

        # 提取所有图片的特征
        feature_list = []
        valid_indices = []  # 记录成功提取特征的图片在列表中的索引

        for i, image_path in enumerate(tqdm(self.image_paths, desc="提取特征")):
            try:
                # 确保图片文件确实存在于磁盘上
                if not os.path.exists(image_path):
                    print(f"警告: 图片文件不存在，已跳过: {image_path}")
                    continue

                features = self.extract_features(image_path)

                # 检查特征是否有效
                feature_norm = np.linalg.norm(features)
                if feature_norm < 1e-10:
                    print(f"警告: 图片 {image_path} 的特征范数过低: {feature_norm:.6f}")
                    continue

                feature_list.append(features)
                valid_indices.append(i)  # 标记这个索引是有效的

            except Exception as e:
                print(f"提取图片 {image_path} 特征失败: {e}")

        # 如果没有成功提取到任何特征
        if not feature_list:
            print("未能从任何图片中提取到有效特征。")
            self._reset_index()
            return

        # --- 重要：根据有效特征过滤路径和ID列表 ---
        # 因为有些图片可能提取特征失败，我们需要确保
        # self.features, self.image_paths, self.image_ids 三者的顺序和数量是完全对应的
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.image_ids = [self.image_ids[i] for i in valid_indices]

        # 合并特征向量 (n_images, feature_dim)
        self.features = np.vstack(feature_list)

        # 构建FAISS索引前检查特征
        if len(feature_list) > 0:
            all_features = np.vstack(feature_list)
            norms = np.linalg.norm(all_features, axis=1)
            print(f"特征范数统计: 最小={np.min(norms):.6f}, 最大={np.max(norms):.6f}, 平均={np.mean(norms):.6f}")

            # 检查是否有异常的特征向量
            abnormal_indices = np.where((norms < 0.9) | (norms > 1.1))[0]
            if len(abnormal_indices) > 0:
                print(f"警告: 发现 {len(abnormal_indices)} 个异常特征向量（范数不在0.9-1.1范围内）")
                for idx in abnormal_indices[:5]:  # 只打印前5个
                    print(f"  索引 {idx}: 范数={norms[idx]:.6f}, 图片={self.image_paths[valid_indices[idx]]}")

        # 构建FAISS索引
        self.build_index()

        print(f"特征索引构建完成。共 {len(self.features)} 张图片成功入库。")
        print(f"特征维度: {self.features.shape[1]}")

    def _reset_index(self):
        """重置索引相关属性（辅助方法）"""
        self.features = None
        self.index = None
        self.image_paths = []
        self.image_ids = []

    def build_index(self):
        """
        构建FAISS索引用于快速相似度搜索。
        这个方法现在依赖于 self.features 是否为None。
        """
        if self.features is None:
            print("没有特征可以构建索引。")
            self.index = None
            return

        # 获取特征维度
        feature_dim = self.features.shape[1]

        # 创建Flat索引（适合小规模图片库，精确匹配）
        # 对于大规模图片库，可以使用IVF等索引加速
        self.index = faiss.IndexFlatL2(feature_dim)

        # 添加特征到索引
        self.index.add(self.features)
        print(f"FAISS索引已构建，包含 {self.index.ntotal} 个特征向量。")

    def find_most_similar(self, query_image, top_k=1):
        """
        查找与查询图片最相似的图片
        """
        # 确保特征索引已加载
        if self.index is None:
            print("索引为空，重新加载特征...")
            self.load_image_features_from_db()
            if self.index is None:
                print("加载特征后索引仍为空")
                return []

        # 检查是否有可用的特征
        if self.features is None or len(self.features) == 0:
            print("没有可用的特征向量")
            return []

        # 提取查询图片的特征
        try:
            print("开始提取查询图片特征...")
            query_features = self.extract_features(query_image)
            print(f"查询特征形状: {query_features.shape}")
        except Exception as e:
            print(f"查询图片特征提取失败: {str(e)}")
            return []

        # 搜索相似图片
        try:
            top_k = min(top_k, len(self.features))
            print(f"搜索 top_k: {top_k}, 特征库大小: {len(self.features)}")

            distances, indices = self.index.search(query_features, top_k)
            print(f"FAISS 搜索结果 - 距离: {distances}, 索引: {indices}")

            # 添加详细的调试信息
            print(f"查询特征范数: {np.linalg.norm(query_features)}")
            print(
                f"特征库特征范数范围: {np.min(np.linalg.norm(self.features, axis=1)):.4f} - {np.max(np.linalg.norm(self.features, axis=1)):.4f}")

            if len(distances[0]) > 0:
                print(f"最小距离: {np.min(distances[0]):.6f}, 最大距离: {np.max(distances[0]):.6f}")
                print(f"最小距离的平方根: {np.sqrt(np.min(distances[0])):.6f}, 最大距离的平方根: {np.sqrt(np.max(distances[0])):.6f}")

            # 处理结果
            results = []
            for i in range(len(indices[0])):
                idx_in_features = indices[0][i]
                distance = distances[0][i]

                print(f"结果 {i + 1}: 索引={idx_in_features}, 距离={distance:.6f},距离平方根={np.sqrt(distance):.6f}")

                # 验证索引有效性
                if idx_in_features >= len(self.image_paths):
                    print(f"无效索引: {idx_in_features}, 最大索引: {len(self.image_paths) - 1}")
                    continue

                similar_image_path = self.image_paths[idx_in_features]
                similar_image_id = self.image_ids[idx_in_features]

                # 计算余弦相似度百分比
                similarity = 1 - (distance / 2)
                similarity_percent = max(0, min(100, similarity * 100))  # 限制在0-100之间

                # 详细打印相似度计算过程
                print(f"  - 原始相似度: {similarity:.6f}")
                print(f"  - 百分比: {similarity_percent:.2f}%")

                # 检查是否为0或异常值
                if similarity_percent < 1:  # 低于1%
                    print(f"  ⚠️ 警告: 相似度异常低 ({similarity_percent:.2f}%)")
                    # 检查特征向量是否正常
                    if idx_in_features < len(self.features):
                        print(f"  - 特征库特征范数: {np.linalg.norm(self.features[idx_in_features]):.6f}")
                        print(f"  - 查询特征范数: {np.linalg.norm(query_features[0]):.6f}")
                        # 手动计算余弦相似度
                        query_norm = query_features[0] / np.linalg.norm(query_features[0])
                        db_norm = self.features[idx_in_features] / np.linalg.norm(self.features[idx_in_features])
                        manual_similarity = np.dot(query_norm, db_norm)
                        print(f"  - 手动计算相似度: {manual_similarity:.6f}, 百分比: {manual_similarity * 100:.2f}%")

                similarity_percent = float(similarity_percent)

                results.append({
                    "image_id": int(similar_image_id),  # 确保是 int
                    "path": str(similar_image_path),  # 确保是 str
                    "similarity_percent": similarity_percent
                })

            print(f"最终结果数量: {len(results)}")
            print(f"搜索结果: {results}")
            return results

        except Exception as e:
            print(f"FAISS 搜索失败: {str(e)}")
            return []



