"""
Week 11b 无学习基线在两种评估协议下的表现（秒级完成）
══════════════════════════════════════════════════════════════════════
目的：量化「随机负采样」协议本身白送了多少 AUC。

三种「不需要学习」的判边函数：
    degree-product  deg(u) * deg(v)          ← 只有一个标量，纯度数
    CN              共同邻居数
    AA              Adamic-Adar
    Jaccard         共同邻居 / 邻居并集

对每种函数，在同一批 10 个边划分上分别计算
    A 协议：测试负样本 = 随机不存在的边
    B 协议：测试负样本 = 度数分层匹配的不存在的边（src/evaluation.py）
然后报告 AUC 的均值 ± 标准差。

运行: uv run python week11b_heuristic_protocol.py [cora|citeseer]
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.graph_utils import load_cora, load_citeseer, split_edges
from src.similarity import compute_scores
from src.evaluation import sample_degree_matched
from sklearn.metrics import roc_auc_score

OUT = "results/repeated"
SEEDS = list(range(1, 11))
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cora"

HALF_LIFE = None


def main():
    G = load_cora() if DATASET == "cora" else load_citeseer()
    G = G[0] if isinstance(G, tuple) else G
    res = {name: {"A": [], "B": []}
           for name in ("degree-product", "CN", "Jaccard", "AA")}

    for seed in SEEDS:
        G_train, tp, tn_rand = split_edges(G, 0.2, seed=seed)
        be = sample_degree_matched(G_train, tp, seed=seed + 1000)
        deg = dict(G_train.degree())

        for proto, tn in (("A", tn_rand), ("B", be)):
            y = np.array([1] * len(tp) + [0] * len(tn))
            pairs = tp + tn

            dp = np.array([deg[u] * deg[v] for u, v in pairs], float)
            res["degree-product"][proto].append(roc_auc_score(y, dp))
            for meth, key in (("CN", "cn"), ("Jaccard", "jaccard"), ("AA", "aa")):
                s = compute_scores(G_train, pairs, method=key)
                res[meth][proto].append(roc_auc_score(y, s))
        print(f"  seed={seed} done", flush=True)

    print(f"\n📊 {DATASET.upper()} 无学习判边函数：A(随机负样本) vs B(度数匹配负样本)", flush=True)
    print(f"  {'判边函数':16s} {'A 随机负样本':>20s} {'B 度数匹配':>20s} {'Δ':>9s}", flush=True)
    for name, d in res.items():
        a, b = np.array(d["A"]), np.array(d["B"])
        print(f"  {name:16s} {a.mean():.4f} ± {a.std(ddof=1):.4f}   "
              f"{b.mean():.4f} ± {b.std(ddof=1):.4f}   {b.mean()-a.mean():+.4f}", flush=True)

    path = os.path.join(OUT, f"{DATASET}_heuristics_ab.json")
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"💾 {path}", flush=True)


if __name__ == "__main__":
    main()
