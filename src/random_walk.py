"""
random_walk.py - 随机游走模块
══════════════════════════════
Week 2 核心代码：在图上"走一走"

随机游走的直觉：
────────────────
想象你站在一个城市的十字路口（= 图上的一个节点）。
你随机选一个方向走到下一个路口（= 沿着边走到邻居节点）。
到了新路口，你再随机选一个方向...如此重复。

走了 10 步后，你记录下经过的路口序列：[A, B, D, B, E, F, ...]

关键洞察：
  - 如果两个路口经常一起出现在你的路线中，
    说明它们在城市里"距离很近" / "属于同一个片区"。
  - 这条路线就编码了城市（图）的结构信息！

这就是 DeepWalk 的核心思想：
  1. 在图上做很多次随机游走，收集"路径"
  2. 把路径当成"句子"，节点当成"词"
  3. 用 Word2Vec 训练 → 得到每个节点的向量表示
"""

import random
import networkx as nx
import numpy as np
from typing import Optional


def uniform_random_walk(
    G: nx.Graph,
    start_node,
    walk_length: int = 10,
    rng: Optional[random.Random] = None,
) -> list:
    """
    从 start_node 出发，做一次均匀随机游走。

    "均匀"的意思：在每一步，走到每个邻居的概率都相等。
    如果当前节点有 4 个邻居，走到每个邻居的概率都是 1/4。

    参数:
        G: 图
        start_node: 起始节点
        walk_length: 游走步数
        rng: 随机数生成器（用于可复现性）

    返回:
        [node_0, node_1, node_2, ...] 长度为 walk_length 的节点序列

    示例:
        walk = uniform_random_walk(G, start_node=35, walk_length=10)
        # 可能得到 [35, 1033, 456, 1033, 789, ...]
        # 注意：可能会"回头"走，这是正常的
    """
    if rng is None:
        rng = random.Random()

    walk = [start_node]

    for _ in range(walk_length - 1):
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))

        if len(neighbors) == 0:
            # 走到了孤立节点（死胡同），停下来
            break

        # 均匀随机选一个邻居
        next_node = rng.choice(neighbors)
        walk.append(next_node)

    return walk


def generate_walks(
    G: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 80,
    seed: int = 42,
) -> list[list]:
    """
    对图中每个节点做 num_walks 次随机游走。

    这是 DeepWalk 的第一步：生成大量的随机游走序列（"语料库"）。

    参数:
        G: 图
        num_walks: 每个节点生成几条游走
        walk_length: 每条游走走几步
        seed: 随机种子

    返回:
        list of walks，每条 walk 是一个节点 ID 列表
        总共有 num_walks * |V| 条游走
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    walks = []

    for i in range(num_walks):
        rng.shuffle(nodes)  # 每轮打乱节点顺序
        for node in nodes:
            walk = uniform_random_walk(G, node, walk_length, rng)
            walks.append(walk)

        # 进度提示
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  🚶 第 {i+1}/{num_walks} 轮游走完成 "
                  f"（已生成 {len(walks)} 条路径）")

    return walks


def demo_random_walk(G: nx.Graph, start_node=None, walk_length: int = 15, seed: int = 42):
    """
    演示一次随机游走的过程，帮助理解随机游走在做什么。
    """
    if start_node is None:
        # 选择度最高的节点作为起点
        start_node = max(G.degree(), key=lambda x: x[1])[0]

    print(f"\n{'─'*60}")
    print(f"🚶 随机游走演示")
    print(f"{'─'*60}")
    print(f"  起点: 节点 {start_node}（度 = {G.degree(start_node)}）")
    print(f"  步数: {walk_length}")
    print()

    rng = random.Random(seed)
    walk = uniform_random_walk(G, start_node, walk_length, rng)

    # 可视化游走过程
    print(f"  游走路径:")
    print(f"  ", end="")
    for i, node in enumerate(walk):
        if i > 0:
            print(f" → ", end="")
        print(f"[{node}]", end="")
        if (i + 1) % 6 == 0 and i < len(walk) - 1:
            print(f"\n  ", end="")
    print()

    # 分析这条路径
    unique_nodes = set(walk)
    visit_counts = {}
    for node in walk:
        visit_counts[node] = visit_counts.get(node, 0) + 1

    print(f"\n  📊 路径分析:")
    print(f"     经过的不同节点数: {len(unique_nodes)} / {walk_length}")
    print(f"     重复访问的节点:")
    for node, count in sorted(visit_counts.items(), key=lambda x: -x[1]):
        if count > 1:
            print(f"       节点 {node}: 访问了 {count} 次")

    # 关键洞察
    print(f"\n  💡 关键洞察:")
    print(f"     - 经常一起出现在路径中的节点，在图中'结构相似'")
    print(f"     - 这条路径可以看成一个'句子'，节点是'词'")
    print(f"     - DeepWalk 会生成成千上万条这样的路径")
    print(f"     - 然后用 Word2Vec 从这些'句子'中学习每个'词'的含义")

    return walk


# ============================================================
# Node2Vec 有偏随机游走 (二阶马尔可夫链)
# ============================================================

def node2vec_random_walk(
    G: nx.Graph,
    start_node,
    walk_length: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    rng: Optional[random.Random] = None,
) -> list:
    """
    基于 Node2Vec 参数 p 和 q 进行二阶有偏随机游走。

    参数:
        G: 图
        start_node: 起始节点
        walk_length: 游走长度
        p: 返回参数 (return parameter)，控制回头的概率
        q: 入出参数 (in-out parameter)，控制向外探索的概率
        rng: 随机数生成器

    返回:
        [node_0, node_1, ...] 游走路径
    """
    if rng is None:
        rng = random.Random()

    walk = [start_node]

    # 第一步：没有“上一步”，只能均匀随机选择邻居
    neighbors = list(G.neighbors(start_node))
    if len(neighbors) == 0:
        return walk
    walk.append(rng.choice(neighbors))

    # 后续步骤：根据“上一步”和“当前步”偏置概率选择下一步
    for _ in range(walk_length - 2):
        curr = walk[-1]
        prev = walk[-2]
        
        curr_neighbors = list(G.neighbors(curr))
        if len(curr_neighbors) == 0:
            break
            
        # 计算每个候选邻居 x 的转移概率
        probabilities = []
        for x in curr_neighbors:
            if x == prev:
                # 1. 回头路 (d_tx = 0)
                bias = 1.0 / p
            elif G.has_edge(prev, x):
                # 2. 原地打转 (d_tx = 1)
                bias = 1.0
            else:
                # 3. 向外扩张 (d_tx = 2)
                bias = 1.0 / q
            
            # 这里对于无权图权重是 1.0，如果是加权图则乘以边的权重 G[curr][x].get('weight', 1.0)
            weight = G[curr][x].get('weight', 1.0)
            probabilities.append(bias * weight)
            
        # 归一化并进行有偏随机选择
        prob_sum = sum(probabilities)
        if prob_sum == 0:
            break
        normalized_probs = [prob / prob_sum for prob in probabilities]
        
        # 采用 random.choices 进行概率采样
        next_node = rng.choices(curr_neighbors, weights=normalized_probs, k=1)[0]
        walk.append(next_node)
        
    return walk


def generate_node2vec_walks(
    G: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 80,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
) -> list[list]:
    """
    生成 Node2Vec 随机游走数据集。
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    walks = []

    for i in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walk = node2vec_random_walk(G, node, walk_length, p, q, rng)
            walks.append(walk)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  🚶 第 {i+1}/{num_walks} 轮 Node2Vec 有偏游走完成 (p={p}, q={q})")

    return walks


# ============================================================
# Jaccard 加权随机游走 (创新算法)
# ============================================================

def jaccard_weighted_random_walk(
    G: nx.Graph,
    start_node,
    walk_length: int = 10,
    beta: float = 0.1,
    rng: Optional[random.Random] = None,
) -> list:
    """
    从 start_node 出发，做一次基于 Jaccard 相似度加权的随机游走。

    转移概率公式:
        P'(v | u) = [Jaccard(u, v) + beta] / sum_{w in N(u)} [Jaccard(u, w) + beta]

    参数:
        G: 图
        start_node: 起始节点
        walk_length: 游走长度
        beta: 平滑因子，防止被局限在局部，并给 Jaccard=0 的节点探索机会
        rng: 随机数生成器

    返回:
        walk: [node_0, node_1, ...] 路径序列
    """
    if rng is None:
        rng = random.Random()

    walk = [start_node]

    for _ in range(walk_length - 1):
        curr = walk[-1]
        curr_neighbors = list(G.neighbors(curr))
        
        if len(curr_neighbors) == 0:
            break
            
        # 预先获取当前节点的邻居集合，以加速 Jaccard 交并集计算
        neighbors_curr_set = set(curr_neighbors)
        
        # 计算当前节点到所有邻居的转移权重
        weights = []
        for neighbor in curr_neighbors:
            # 快速就地计算 Jaccard
            neighbors_neighbor_set = set(G.neighbors(neighbor))
            intersection = len(neighbors_curr_set & neighbors_neighbor_set)
            union = len(neighbors_curr_set | neighbors_neighbor_set)
            
            jaccard = intersection / union if union > 0 else 0.0
            
            # 加入平滑因子 beta
            weight = jaccard + beta
            weights.append(weight)
            
        weight_sum = sum(weights)
        if weight_sum == 0:
            break
            
        normalized_probs = [w / weight_sum for w in weights]
        
        # 概率采样
        next_node = rng.choices(curr_neighbors, weights=normalized_probs, k=1)[0]
        walk.append(next_node)
        
    return walk


def generate_jaccard_weighted_walks(
    G: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 80,
    beta: float = 0.1,
    seed: int = 42,
) -> list[list]:
    """
    生成 Jaccard 加权随机游走语料库。
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    walks = []

    for i in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walk = jaccard_weighted_random_walk(G, node, walk_length, beta, rng)
            walks.append(walk)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  🚶 第 {i+1}/{num_walks} 轮 Jaccard 加权游走完成 (beta={beta})")

    return walks


