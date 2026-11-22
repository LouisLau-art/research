"""
advanced_walk.py - 更一般化的「结构相似度加权随机游走」实现
══════════════════════════════════════════════════════════════════════
Week 9 新增：为中期答辩后的算法改进预留的一个可插拔加权框架。

动机
────
开题方案中的线性加权形式为

    P(v → x) ∝ α(t, x) · w(v, x) · (β + Jaccard(v, x))

前期实验（10 次随机划分 + 配对检验）显示：该线性形式相对标准 DeepWalk
的提升 Δ=+0.0044，p=0.44，不具统计显著性。原因推测有两点：
  1. 线性偏置使相邻邻居之间的权差距被 β 抹平（J 通常在 0~0.3，β=0.2 时
     (J+β) 的相对差异很小），转移分布几乎退化为均匀游走；
  2. 只用 Jaccard 一种相似度，未纳入度数惩罚（AA / RA）。

因此本模块提供一个带「锐化」能力的通用加权游走：
    P(v → x) ∝ α(t, x) · ( s(v, x) + β )^γ
其中 s 可取 Jaccard / Adamic-Adar / Resource-Allocation，γ 控制锐化程度。
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from gensim.models import Word2Vec


# ══════════════════════════════════════════════════════════════════════
# 1. 相似度函数（带缓存）
# ══════════════════════════════════════════════════════════════════════

class SimilarityCache:
    """为无向边缓存相似度分数，避免游走时重复计算。"""

    def __init__(self, G: nx.Graph, kind: str = "jaccard"):
        self.G = G
        self.kind = kind
        self._cache: Dict[Tuple, float] = {}
        self._nbrs = {n: set(G.neighbors(n)) for n in G.nodes()}

    def get(self, u, v) -> float:
        key = (u, v) if u <= v else (v, u)
        if key in self._cache:
            return self._cache[key]
        val = self._compute(u, v)
        self._cache[key] = val
        return val

    def _compute(self, u, v) -> float:
        nu, nv = self._nbrs[u], self._nbrs[v]
        inter = nu & nv
        if self.kind == "jaccard":
            union = nu | nv
            return len(inter) / len(union) if union else 0.0
        if self.kind == "aa":
            # Adamic-Adar：共同邻居按 1/log(deg) 加权（度数越小越重要）
            s = 0.0
            for w in inter:
                d = len(self._nbrs[w])
                if d > 1:
                    s += 1.0 / np.log(d)
            return s
        if self.kind == "ra":
            # Resource Allocation：共同邻居按 1/deg 加权
            s = 0.0
            for w in inter:
                d = len(self._nbrs[w])
                if d > 0:
                    s += 1.0 / d
            return s
        raise ValueError(f"unknown similarity kind: {self.kind}")


def normalize_by_max(G: nx.Graph, cache: SimilarityCache) -> None:
    """
    AA/RA 的取值范围不是 [0,1]，直接和 β 相加量纲不匹配。
    这里把每条边上的分数除以该分数的全局最大值，归一到 [0,1]，
    保证与 Jaccard 的加权口径可比。
    """
    mx = 0.0
    for u, v in G.edges():
        mx = max(mx, cache.get(u, v))
    if mx <= 0:
        return
    frozen = dict(cache._cache)
    for k, val in frozen.items():
        cache._cache[k] = val / mx


# ══════════════════════════════════════════════════════════════════════
# 2. 通用的加权随机游走
# ══════════════════════════════════════════════════════════════════════

def weighted_random_walk(
    G: nx.Graph,
    start,
    walk_length: int,
    cache: SimilarityCache,
    beta: float = 0.2,
    gamma: float = 1.0,
    p: Optional[float] = None,
    q: Optional[float] = None,
    rng: Optional[random.Random] = None,
) -> List:
    """
    单条加权随机游走。

    转移权重：
        w(v → x) = alpha(t, x) * ( s(v, x) + beta ) ^ gamma

    - s(v, x)      : 结构相似度（jaccard / aa / ra）
    - beta         : 平滑项，保证 s=0 时仍有非零概率（鲁棒性）
    - gamma        : 锐化指数，gamma 越大越偏向高相似度邻居
    - p, q         : 若给定，则叠加 Node2Vec 的返回/外出偏置 alpha(t, x)；
                     否则就是纯相似度加权（DeepWalk 式无记忆游走）
    """
    if rng is None:
        rng = random.Random()

    walk = [start]
    for _ in range(walk_length - 1):
        cur = walk[-1]
        nbrs = list(G.neighbors(cur))
        if not nbrs:
            break

        prev = walk[-2] if len(walk) > 1 else None
        weights = []
        for x in nbrs:
            w = (cache.get(cur, x) + beta) ** gamma
            if p is not None and q is not None and prev is not None:
                if x == prev:
                    w *= 1.0 / p          # 返回上一步
                elif G.has_edge(x, prev):
                    w *= 1.0              # 与上一步相邻 → 广度优先
                else:
                    w *= 1.0 / q          # 离开上一步邻域 → 深度优先
            weights.append(w)

        total = sum(weights)
        if total <= 0:
            break
        # 轮盘赌采样
        r = rng.random() * total
        acc = 0.0
        for x, w in zip(nbrs, weights):
            acc += w
            if r <= acc:
                walk.append(x)
                break
        else:
            walk.append(nbrs[-1])
    return walk


def generate_weighted_walks(
    G: nx.Graph,
    cache: SimilarityCache,
    num_walks: int = 10,
    walk_length: int = 40,
    beta: float = 0.2,
    gamma: float = 1.0,
    p: Optional[float] = None,
    q: Optional[float] = None,
    seed: int = 42,
) -> List[List]:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walks.append(weighted_random_walk(
                G, node, walk_length, cache, beta=beta, gamma=gamma,
                p=p, q=q, rng=rng))
    return walks


# ══════════════════════════════════════════════════════════════════════
# 3. 把游走序列喂给 Skip-gram
# ══════════════════════════════════════════════════════════════════════

class WeightedWalkEmbedding:
    """
    通用加权游走 + Skip-gram 的节点表示学习器。

    参数:
        sim_kind : 'jaccard' | 'aa' | 'ra'
        beta     : 平滑项
        gamma    : 锐化指数
        p, q     : 若给定则叠加 Node2Vec 的偏置（"Jaccard-weighted Node2Vec"）
    """

    def __init__(self, dimensions=64, walk_length=40, num_walks=10,
                 sim_kind="jaccard", beta=0.2, gamma=1.0,
                 p=None, q=None, window_size=5, workers=2, seed=42):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.sim_kind = sim_kind
        self.beta = beta
        self.gamma = gamma
        self.p = p
        self.q = q
        self.window_size = window_size
        self.workers = workers
        self.seed = seed
        self.embeddings: Dict = {}

    def fit(self, G: nx.Graph) -> Dict:
        cache = SimilarityCache(G, kind=self.sim_kind)
        if self.sim_kind in ("aa", "ra"):
            normalize_by_max(G, cache)

        walks = generate_weighted_walks(
            G, cache, num_walks=self.num_walks, walk_length=self.walk_length,
            beta=self.beta, gamma=self.gamma, p=self.p, q=self.q, seed=self.seed)

        str_walks = [[str(n) for n in w] for w in walks]
        model = Word2Vec(
            sentences=str_walks, vector_size=self.dimensions,
            window=self.window_size, min_count=0, sg=1, hs=0,
            negative=5, workers=self.workers, seed=self.seed)
        self.embeddings = {n: model.wv[n] for n in model.wv.index_to_key}
        return self.embeddings
