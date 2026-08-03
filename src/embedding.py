"""
embedding.py - 图表示学习（嵌入）算法模块
==========================================
Week 4-5 核心代码：实现 DeepWalk 与 Node2Vec 节点嵌入算法

本模块将图节点序列（随机游走路径）通过 gensim Word2Vec 转换为连续向量表示。
"""

import os
import random
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from typing import Dict, List, Optional
from src.random_walk import uniform_random_walk, generate_walks, generate_node2vec_walks, generate_jaccard_weighted_walks


class DeepWalk:
    """
    DeepWalk 算法实现 (Perozzi et al., KDD 2014)

    算法步骤:
      1. 对图中的每个节点，生成若干条均匀随机游走路径。
      2. 将游走路径视为句子（节点ID为单词），使用 Word2Vec (Skip-gram) 训练得到节点向量。
    """
    def __init__(
        self,
        dimensions: int = 64,
        walk_length: int = 40,
        num_walks: int = 10,
        window_size: int = 5,
        workers: int = 2,
        seed: int = 42
    ):
        """
        初始化 DeepWalk 参数。

        参数:
            dimensions: 节点嵌入向量的维度 (d)
            walk_length: 每次随机游走的步数 (l)
            num_walks: 每个节点作为起点生成的随机游走次数 (r)
            window_size: Skip-gram 模型的滑动窗口大小 (w)
            workers: 训练的线程数
            seed: 随机种子
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.workers = workers
        self.seed = seed
        self.embeddings: Dict[str, np.ndarray] = {}

    def fit(self, G: nx.Graph) -> Dict[str, np.ndarray]:
        """
        在给定的图上训练 DeepWalk 模型并得到节点嵌入向量。

        参数:
            G: NetworkX 图对象

        返回:
            embeddings: {node_str: embedding_vector} 的字典
        """
        print(f"🎬 开始训练 DeepWalk (维度={self.dimensions}, 长度={self.walk_length}, 每节点游走={self.num_walks})")
        
        # 1. 生成随机游走序列
        walks = generate_walks(
            G,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            seed=self.seed
        )
        
        # 将节点ID转化为字符串，以供 Word2Vec 词表使用
        str_walks = [[str(node) for node in walk] for walk in walks]

        # 2. 训练 Word2Vec Skip-gram 模型
        print("⚡ 正在通过 Word2Vec Skip-gram 学习节点特征向量...")
        model = Word2Vec(
            sentences=str_walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,              # 1 = Skip-gram, 0 = CBOW
            hs=0,              # 0 = 负采样 (Negative Sampling), 1 = 分层 Softmax
            negative=5,        # 负采样样本数
            workers=self.workers,
            seed=self.seed
        )
        
        # 将嵌入结果整理成字典形式
        self.embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
        print("✅ DeepWalk 训练完成！")
        return self.embeddings

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        return self.embeddings


class Node2Vec:
    """
    Node2Vec 算法实现 (Grover & Leskovec, KDD 2016)

    算法步骤:
      1. 对图中的每个节点，生成若干条有偏二阶随机游走路径（受 p 和 q 控制）。
      2. 使用 Word2Vec (Skip-gram) 训练得到节点向量。
    """
    def __init__(
        self,
        dimensions: int = 64,
        walk_length: int = 40,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        window_size: int = 5,
        workers: int = 2,
        seed: int = 42
    ):
        """
        初始化 Node2Vec 参数。

        参数:
            dimensions: 节点嵌入向量的维度
            walk_length: 每次随机游走的步数
            num_walks: 每个节点作为起点生成的随机游走次数
            p: 返回参数 (return parameter)，p 越小越倾向于 BFS
            q: 入出参数 (in-out parameter)，q 越小越倾向于 DFS
            window_size: Skip-gram 滑动窗口大小
            workers: 线程数
            seed: 随机种子
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.workers = workers
        self.seed = seed
        self.embeddings: Dict[str, np.ndarray] = {}

    def fit(self, G: nx.Graph) -> Dict[str, np.ndarray]:
        """
        在给定的图上训练 Node2Vec 模型并得到节点嵌入向量。
        """
        print(f"🎬 开始训练 Node2Vec (维度={self.dimensions}, 长度={self.walk_length}, 每节点游走={self.num_walks}, p={self.p}, q={self.q})")
        
        # 1. 生成有偏二阶随机游走序列
        walks = generate_node2vec_walks(
            G,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            p=self.p,
            q=self.q,
            seed=self.seed
        )
        
        # 将节点ID转化为字符串
        str_walks = [[str(node) for node in walk] for walk in walks]

        # 2. 训练 Word2Vec Skip-gram 模型
        print("⚡ 正在通过 Word2Vec Skip-gram 学习节点特征向量...")
        model = Word2Vec(
            sentences=str_walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,
            hs=0,
            negative=5,
            workers=self.workers,
            seed=self.seed
        )
        
        self.embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
        print("✅ Node2Vec 训练完成！")
        return self.embeddings

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        return self.embeddings


class JaccardWeightedDeepWalk:
    """
    基于 Jaccard 相似度加权的 DeepWalk 改进算法 (创新算法)

    算法步骤:
      1. 对图中的每个节点，生成若干条 Jaccard 加权随机游走路径（受 beta 平滑因子控制）。
      2. 将游走路径转换为节点向量表示，通过 Word2Vec Skip-gram 学习。
    """
    def __init__(
        self,
        dimensions: int = 64,
        walk_length: int = 40,
        num_walks: int = 10,
        beta: float = 0.1,
        window_size: int = 5,
        workers: int = 2,
        seed: int = 42
    ):
        """
        初始化 Jaccard 加权 DeepWalk 参数。

        参数:
            dimensions: 节点嵌入向量维度
            walk_length: 游走长度
            num_walks: 每个节点游走次数
            beta: 平滑因子 (beta)
            window_size: 滑动窗口大小
            workers: 线程数
            seed: 随机种子
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.beta = beta
        self.window_size = window_size
        self.workers = workers
        self.seed = seed
        self.embeddings: Dict[str, np.ndarray] = {}

    def fit(self, G: nx.Graph) -> Dict[str, np.ndarray]:
        """
        在给定的图上训练 Jaccard 加权 DeepWalk 模型并得到节点嵌入向量。
        """
        print(f"🎬 开始训练 Jaccard 加权 DeepWalk (维度={self.dimensions}, 长度={self.walk_length}, 每节点游走={self.num_walks}, beta={self.beta})")
        
        # 1. 生成 Jaccard 加权随机游走序列
        walks = generate_jaccard_weighted_walks(
            G,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            beta=self.beta,
            seed=self.seed
        )
        
        # 将节点ID转化为字符串
        str_walks = [[str(node) for node in walk] for walk in walks]

        # 2. 训练 Word2Vec Skip-gram 模型
        print("⚡ 正在通过 Word2Vec Skip-gram 学习节点特征向量...")
        model = Word2Vec(
            sentences=str_walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,
            hs=0,
            negative=5,
            workers=self.workers,
            seed=self.seed
        )
        
        self.embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
        print("✅ Jaccard 加权 DeepWalk 训练完成！")
        return self.embeddings

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        return self.embeddings


