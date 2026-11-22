"""
evaluation.py — 修正版的链路预测评估协议
══════════════════════════════════════════════════════════════════════
【为什么需要这个模块】

Week 9-10 的实验暴露出评估协议本身的两个缺陷，它们会系统性地
把「与方法质量无关的信号」算进 AUC：

缺陷 1 — 向量模长捷径（norm shortcut）
    Hadamard 边特征是 u_i * v_i 逐维相乘。如果 u、v 的向量模长都大，
    特征的整体幅值就大，逻辑回归可以直接把「模长大 = 正样本」学成判据。
    而 Word2Vec 向量模长与节点出现频次（≈ 度数）高度相关，
    于是分类器实际上在用「度数」判边。

缺陷 2 — 随机负样本的度偏好（degree bias）
    Liben-Nowell 的随机负采样从全部不存在的节点对里均匀抽。
    但真实边（含被藏起测试边）天然更可能连接高度节点（优先连接效应）：
        Citeseer：正测试边平均度积 27.0，负测试边 5.3
        Cora    ：正测试边平均度积 42.3，负测试边 9.0
    因此任何与度数相关的打分都能拿到偏高的 AUC。仅用「度积」一个特征
    做逻辑回归，Citeseer 就能拿 0.570、Cora 能拿 0.647。

【本模块提供两项修正】

    normalize_embeddings(emb)   把每个向量归一化到单位长度，消除模长尺度
    sample_degree_matched(...)  按度数分层采样负样本，使负样本的度积分布
                                与正样本匹配，消除优先连接偏置

两者结合 = 「修正协议」。Week 11 的脚本会在同一批边划分上
分别跑原协议与修正协议，比较方法排名是否稳定。
"""
from typing import Dict, List, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════
# 修正 1：向量单位归一化
# ══════════════════════════════════════════════════════════════════

def normalize_embeddings(emb: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """把每个节点向量缩放到单位长度。Hadamard 积的幅值就不再随度数变化。"""
    out = {}
    for k, v in emb.items():
        n = np.linalg.norm(v)
        out[k] = v / n if n > 0 else v
    return out


# ══════════════════════════════════════════════════════════════════
# 修正 2：按度数分层的负采样
# ══════════════════════════════════════════════════════════════════

def _degree_buckets(degrees: Dict, n_buckets: int = 60) -> Dict[int, np.ndarray]:
    """把节点按度数排序后切成等频桶，返回 {bucket_id: node_array}。"""
    # 次要键用 str(node)：度数相同的节点若按原始插入顺序排，字符串节点会
    # 受 PYTHONHASHSEED 影响，导致负采样跨进程不一致。
    nodes = sorted(degrees, key=lambda n: (degrees[n], str(n)))
    buckets = np.array_split(np.array(nodes, dtype=object), n_buckets)
    return {i: b for i, b in enumerate(buckets) if len(b) > 0}


def _bucket_of(degrees: Dict, buckets: Dict[int, np.ndarray]) -> Dict:
    m = {}
    for bid, b in buckets.items():
        for n in b:
            m[n] = bid
    return m


def sample_degree_matched(
    G_train,
    positives: List[Tuple],
    n_samples: int = None,
    seed: int = 42,
    max_tries_per_sample: int = 200,
) -> List[Tuple]:
    """
    对每一条正样本边 (u, v)，从「与 u 同度数桶」「与 v 同度数桶」里各随机取一个节点
    u'、v'，拼成候选负样本；校验 (u', v') 在训练图中确实不存在、且不与已采样本重复。
    这样负样本的度积分布与正样本大幅对齐（n_buckets=60 时，Cora 上负样本平均
    度积由 14.8 提升到 38.6，正样本目标 54.9；残余差距源自 hub–hub 的不存在边
    本身就稀少，属该协议的已知局限）。

    参数:
        G_train  : 训练图（测试边已被移除）
        positives: 需要匹配的正样本边列表（训练时用训练正样本，测试时用测试正样本）
        n_samples: 采样条数，默认 = len(positives)
    """
    if n_samples is None:
        n_samples = len(positives)

    degrees = dict(G_train.degree())
    buckets = _degree_buckets(degrees, n_buckets=60)
    bucket_of = _bucket_of(degrees, buckets)
    rng = np.random.RandomState(seed)
    exist = {frozenset(e) for e in G_train.edges()}

    negatives, seen = [], set()
    src = list(positives)
    idx = 0
    tries = 0

    while len(negatives) < n_samples:
        u, v = src[idx % len(src)]
        idx += 1
        tries += 1
        if tries > n_samples * max_tries_per_sample:
            break  # 兜底：采样空间太小时不要死循环

        bu, bv = bucket_of[u], bucket_of[v]
        # 度数桶太小时退化为全图随机
        cu = rng.choice(buckets[bu]) if len(buckets[bu]) > 1 else rng.choice(list(degrees))
        cv = rng.choice(buckets[bv]) if len(buckets[bv]) > 1 else rng.choice(list(degrees))
        if cu == cv:
            continue
        key = frozenset((cu, cv))
        if key in exist or key in seen:
            continue
        seen.add(key)
        negatives.append((cu, cv))

    return negatives


def negative_degree_stats(G, negatives: List[Tuple]) -> float:
    """返回负样本的平均度积，用于校验分层是否对齐。"""
    deg = dict(G.degree())
    return float(np.mean([deg[u] * deg[v] for u, v in negatives]))
