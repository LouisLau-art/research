"""
similarity.py - 经典链路预测相似度指标
════════════════════════════════════════
Week 2 核心代码：实现四种经典的拓扑相似度指标

这些指标的核心思想只有一个：
    两个节点的"邻居"越像，它们之间越可能有边。

区别在于"怎么度量像不像"：
    - Common Neighbors: 数共同好友的个数
    - Jaccard:          共同好友 ÷ 所有好友的并集（归一化）
    - Adamic-Adar:      给"稀有"的共同好友更高权重
    - Resource Allocation: 类似 AA，但惩罚更狠
"""

import networkx as nx
import numpy as np
from typing import Sequence
import math


# ============================================================
# 四种经典相似度指标
# ============================================================

def common_neighbors_score(G: nx.Graph, u, v) -> float:
    """
    Common Neighbors (CN) 指标

    CN(u, v) = |N(u) ∩ N(v)|

    直觉：你和某人有 10 个共同好友 vs 0 个，
          哪种情况你们更可能成为好友？

    优点：简单直接
    缺点：没考虑节点度数的影响（大 V 和谁都有共同好友）
    """
    return len(list(nx.common_neighbors(G, u, v)))


def jaccard_score(G: nx.Graph, u, v) -> float:
    """
    Jaccard Coefficient 指标

    J(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

    直觉：把共同好友数"归一化"了。
          小明(3个好友)和小红(3个好友)有2个共同好友 → J = 0.50
          大V(10000粉丝)和路人(5好友)有2个共同好友 → J = 0.0002

    优点：不受节点度数的影响，更公平
    缺点：如果两个节点都没有邻居，J = 0/0（undefined）

    🔥 这就是你论文要用的核心指标！
    """
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    intersection = neighbors_u & neighbors_v
    union = neighbors_u | neighbors_v
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)


def adamic_adar_score(G: nx.Graph, u, v) -> float:
    """
    Adamic-Adar (AA) 指标

    AA(u, v) = Σ_{w ∈ N(u) ∩ N(v)} 1 / log(|N(w)|)

    直觉：共同好友中，"社恐"（好友少）的权重更高，
          "社牛"（好友多）的权重更低。

          为什么？如果你和我唯一的共同好友是一个只有 3 个朋友的人，
          这说明我们的关系很"特殊"。
          但如果共同好友是一个有 10000 个粉丝的大 V，
          那不说明什么——他和谁都是"共同好友"。

    优点：比 CN 更精细
    缺点：计算量稍大
    """
    score = 0.0
    for w in nx.common_neighbors(G, u, v):
        degree_w = G.degree(w)
        if degree_w > 1:  # log(1) = 0 会导致除零
            score += 1.0 / math.log(degree_w)
    return score


def resource_allocation_score(G: nx.Graph, u, v) -> float:
    """
    Resource Allocation (RA) 指标

    RA(u, v) = Σ_{w ∈ N(u) ∩ N(v)} 1 / |N(w)|

    直觉：想象每个共同好友 w 有 1 单位的"资源"（时间/精力），
          w 把资源平均分给它的所有好友。
          好友少的 w → 分给你的多（它更"专注"在你身上）
          好友多的 w → 分给你的少（它"不够专一"）

    和 AA 的区别：RA 用 1/degree 而不是 1/log(degree)，惩罚更强
    """
    score = 0.0
    for w in nx.common_neighbors(G, u, v):
        degree_w = G.degree(w)
        if degree_w > 0:
            score += 1.0 / degree_w
    return score


# ============================================================
# 批量计算（用于实验）
# ============================================================

def compute_scores(G: nx.Graph, edge_list: Sequence[tuple], method: str = "jaccard") -> np.ndarray:
    """
    对一组节点对批量计算相似度分数。

    参数:
        G: 图
        edge_list: [(u1, v1), (u2, v2), ...] 待评估的节点对
        method: "cn" | "jaccard" | "aa" | "ra"

    返回:
        numpy array of scores, shape = (len(edge_list),)
    """
    method_map = {
        "cn": common_neighbors_score,
        "jaccard": jaccard_score,
        "aa": adamic_adar_score,
        "ra": resource_allocation_score,
    }

    if method not in method_map:
        raise ValueError(f"未知方法: {method}，支持: {list(method_map.keys())}")

    func = method_map[method]
    scores = np.array([func(G, u, v) for u, v in edge_list])
    return scores


# ============================================================
# 指标对比演示
# ============================================================

def demo_compare_metrics(G: nx.Graph, edge_pairs: list[tuple] = None):
    """
    在几对节点上对比四种指标的值，帮助直观理解它们的区别。
    """
    if edge_pairs is None:
        # 默认取前 5 条边
        edge_pairs = list(G.edges())[:5]

    print(f"\n{'─'*70}")
    print(f"📊 四种相似度指标对比")
    print(f"{'─'*70}")
    print(f"{'节点对':>15s}  {'CN':>6s}  {'Jaccard':>8s}  {'AA':>8s}  {'RA':>8s}")
    print(f"{'─'*70}")

    for u, v in edge_pairs:
        cn = common_neighbors_score(G, u, v)
        jac = jaccard_score(G, u, v)
        aa = adamic_adar_score(G, u, v)
        ra = resource_allocation_score(G, u, v)
        print(f"  ({u}, {v}){' '*(10 - len(str(u)) - len(str(v)))}  "
              f"{cn:6.0f}  {jac:8.4f}  {aa:8.4f}  {ra:8.4f}")

    print(f"{'─'*70}")
