"""
link_predictor.py - 链路预测分类器与评估模块
============================================
Week 4 核心代码：将节点嵌入转化为边特征，训练分类器并计算 AUC
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Dict, List, Tuple


def get_edge_features(
    embeddings: Dict[str, np.ndarray],
    edge_list: List[Tuple],
    embedding_size: int,
    operator: str = "hadamard"
) -> np.ndarray:
    """
    根据节点向量，计算边特征。

    参数:
        embeddings: 节点嵌入字典 {node_str: np.ndarray}
        edge_list: 待计算特征的边/节点对列表 [(u1, v1), (u2, v2), ...]
        embedding_size: 节点嵌入的维度 (例如 64)
        operator: 融合算子 ("hadamard" | "l1" | "l2" | "average")

    返回:
        edge_features: 形状为 (len(edge_list), embedding_size) 的特征矩阵
    """
    features = []
    
    for u, v in edge_list:
        u_str, v_str = str(u), str(v)
        
        # 提取节点向量，如果节点在训练中没出现过，则填充为 0
        u_emb = embeddings[u_str] if u_str in embeddings else np.zeros(embedding_size)
        v_emb = embeddings[v_str] if v_str in embeddings else np.zeros(embedding_size)
        
        if operator == "hadamard":
            # 对应位置相乘 (Hadamard product)
            edge_feature = u_emb * v_emb
        elif operator == "l1":
            # 差的绝对值
            edge_feature = np.abs(u_emb - v_emb)
        elif operator == "l2":
            # 差的平方
            edge_feature = (u_emb - v_emb) ** 2
        elif operator == "average":
            # 均值
            edge_feature = 0.5 * (u_emb + v_emb)
        else:
            raise ValueError(f"未知算子: {operator}")
            
        features.append(edge_feature)
        
    return np.array(features)


class LinkPredictor:
    """
    基于机器学习分类器的链路预测评估器。
    通常使用逻辑回归（Logistic Regression）作为底部分类器，因为它的计算开销小，且适合线性决策边界。
    """
    def __init__(self, operator: str = "hadamard"):
        self.operator = operator
        self.clf = LogisticRegression(max_iter=1000, random_state=42)

    def train_and_evaluate(
        self,
        embeddings: Dict[str, np.ndarray],
        embedding_size: int,
        train_pos: List[Tuple],
        train_neg: List[Tuple],
        test_pos: List[Tuple],
        test_neg: List[Tuple]
    ) -> Tuple[float, float]:
        """
        训练二分类器，并在测试集上进行评估。

        参数:
            embeddings: 训练好的节点向量表示
            embedding_size: 向量维度
            train_pos: 训练集正样本（图中的真实边）
            train_neg: 训练集负样本（图中不存在的边）
            test_pos: 测试集正样本（被藏起来的真边）
            test_neg: 测试集负样本（不存在的假边）

        返回:
            auc_score: AUC-ROC 评估分数
            ap_score: AP (Average Precision) 分数
        """
        # 1. 提取训练集和测试集的边特征
        X_train_pos = get_edge_features(embeddings, train_pos, embedding_size, self.operator)
        X_train_neg = get_edge_features(embeddings, train_neg, embedding_size, self.operator)
        X_train = np.vstack([X_train_pos, X_train_neg])
        
        # 标签：正样本为 1，负样本为 0
        y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))
        
        # 测试集特征提取
        X_test_pos = get_edge_features(embeddings, test_pos, embedding_size, self.operator)
        X_test_neg = get_edge_features(embeddings, test_neg, embedding_size, self.operator)
        X_test = np.vstack([X_test_pos, X_test_neg])
        
        y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))
        
        # 2. 训练逻辑回归分类器
        self.clf.fit(X_train, y_train)
        
        # 3. 预测正类的概率
        y_pred_proba = self.clf.predict_proba(X_test)[:, 1]
        
        # 4. 计算评估指标
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # 计算 Average Precision
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ap_score = auc(recall, precision)
        
        return auc_score, ap_score
