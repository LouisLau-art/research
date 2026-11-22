"""
Week 13c —— 模块 4 对平台的输出：给「关系推荐」服务的候选连接排序清单
══════════════════════════════════════════════════════════════════════
模块 4 对平台交付的不是一个 AUC 数字，而是「按分数排序的候选连接」。
本脚本在 Facebook 社交图（4039 点 / 88234 边）上演示这个输出，并同时报告
该输出的实际质量，避免只给一张看起来漂亮的样例表：

  1. 随机抽 200 个查询节点，对每个节点在「训练图中不存在」的候选里取 top-10，
     统计其中有多少是真正被藏起来的边（Precision@5 / Precision@10）
  2. 另取 10 个节点生成 top-5 清单写入 CSV，供平台侧展示

运行: PYTHONHASHSEED=0 ./.venv/bin/python week13c_recommend_sample.py
输出: results/module4_recommend_sample.csv
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.embedding import DeepWalk
from src.evaluation import normalize_embeddings
from src.graph_utils import load_facebook, split_edges
from src.link_predictor import get_edge_features
from sklearn.linear_model import LogisticRegression

DIMS, WL, NW, WIN = 64, 40, 10, 5
SEED = 1
TOPK = 5
EVAL_NODES = 200          # 用于统计 Precision@K 的随机查询节点数
SAMPLE_NODES = 10         # 写入 CSV 的节点数
NEG_FRAC = 0.05           # 训练负样本用非边的一个子集（全量太大）


def main():
    G = load_facebook()
    G = G[0]
    G_train, tp, tn = split_edges(G, 0.2, seed=SEED)
    pos = list(G_train.edges())

    # 训练负样本：非边里随机抽一小部分（非边数量是 10^6 量级）
    from src.graph_utils import all_non_edges
    rng = np.random.RandomState(SEED)
    ne_all = all_non_edges(G_train)
    rng.shuffle(ne_all)
    neg = list(ne_all[:len(pos)])

    print(f"训练边 {len(pos)}  负样本 {len(neg)}  查询节点 {G_train.number_of_nodes()}")
    emb = DeepWalk(dimensions=DIMS, walk_length=WL, num_walks=NW,
                   window_size=WIN, workers=1, seed=SEED).fit(G_train)
    emb = normalize_embeddings(emb)

    X = np.vstack([get_edge_features(emb, pos, DIMS),
                   get_edge_features(emb, neg, DIMS)])
    y = np.array([1] * len(pos) + [0] * len(neg))
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)

    nodes = sorted(G_train.nodes(), key=str)
    query = list(rng.choice(nodes, size=min(EVAL_NODES, len(nodes)), replace=False))

    hits = {5: [], 10: []}
    rows = []
    for u in query:
        cand = [v for v in nodes if v != u and not G_train.has_edge(u, v)]
        if not cand:
            continue
        p = clf.predict_proba(get_edge_features(emb, [(u, v) for v in cand], DIMS))[:, 1]
        order = np.argsort(-p)
        truth = np.array([G.has_edge(u, v) for v in cand])
        for k in (5, 10):
            hits[k].append(float(truth[order[:k]].mean()))

        if len(rows) < SAMPLE_NODES * TOPK:
            for rank, i in enumerate(order[:TOPK], 1):
                v = cand[i]
                rows.append(dict(
                    query_node=int(u), rank=rank, candidate=int(v),
                    score=round(float(p[i]), 4),
                    existing_edge=bool(G.has_edge(u, v)),
                    candidate_degree=int(G_train.degree(v)),
                    common_neighbours=len(list(_common(G_train, u, v))),
                ))

    print(f"\n输出质量（随机 {len(hits[5])} 个查询节点）:")
    for k in (5, 10):
        a = np.array(hits[k])
        print(f"  Precision@{k:<2d} = {a.mean():.4f} ± {a.std(ddof=1):.4f} "
              f"（随机基线约 {(2 * len(pos) / len(ne_all)):.2e}）")

    out = "results/module4_recommend_sample.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_true = sum(1 for r in rows if r["existing_edge"])
    print(f"\n💾 {out}  {len(rows)} 行，其中 {n_true} 行命中真实连接")
    for row in rows[:8]:
        print("  ", row)


def _common(G, u, v):
    """训练图上的共同邻居（有迭代器语义，调用方取 len）。"""
    nu, nv = set(G.neighbors(u)), set(G.neighbors(v))
    return nu & nv


if __name__ == "__main__":
    main()