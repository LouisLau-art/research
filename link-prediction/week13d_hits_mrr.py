"""
Week 13d —— 排序敏感指标 Hits@K / MRR（模块 4 的「推荐」视角）
══════════════════════════════════════════════════════════════════════
AUC 衡量的是「正样本排在负样本前面的概率」，与平台真实的「给我 top-K 候选」
并不等价。本脚本补两个排序指标：

  对每条被藏起的真实边 (u,v)：把它和同一批度数匹配负样本放在一起打分，
  按分数降序求它的名次 rank（1 为最好）
      Hits@10 = P(rank <= 10)
      Hits@50 = P(rank <= 50)
      MRR     = mean(1 / rank)

三种嵌入方法（DeepWalk / Node2Vec-DFS / Ours-Jaccard）× 三个数据集 × 10 划分，
全部在修正协议 B（度数匹配负样本 + 单位归一化向量）下计算。

运行: PYTHONHASHSEED=0 ./.venv/bin/python week13d_hits_mrr.py [cora|citeseer|facebook]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.advanced_walk import WeightedWalkEmbedding
from src.embedding import DeepWalk, Node2Vec
from src.evaluation import normalize_embeddings, sample_degree_matched
from src.graph_utils import load_cora, load_citeseer, load_facebook, split_edges
from src.link_predictor import get_edge_features

OUT = "results/repeated"
SEEDS = list(range(1, 11))
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cora"
DIMS, WL, NW, WIN, WK, BETA = 64, 40, 10, 5, 1, 0.2

METHODS = {
    "DeepWalk":            dict(kind="dw"),
    "Node2Vec (DFS-like)": dict(kind="n2v", p=2.0, q=0.5),
    "Ours Jaccard g=1":    dict(kind="ww", sim_kind="jaccard", gamma=1.0),
}


def load_graph():
    g = {"cora": load_cora, "citeseer": load_citeseer, "facebook": load_facebook}[DATASET]()
    return g[0] if isinstance(g, tuple) else g


def build(cfg, G_train, seed):
    if cfg["kind"] == "dw":
        e = DeepWalk(dimensions=DIMS, walk_length=WL, num_walks=NW, window_size=WIN,
                     workers=WK, seed=seed).fit(G_train)
    elif cfg["kind"] == "n2v":
        e = Node2Vec(dimensions=DIMS, walk_length=WL, num_walks=NW, p=cfg["p"], q=cfg["q"],
                     window_size=WIN, workers=WK, seed=seed).fit(G_train)
    else:
        e = WeightedWalkEmbedding(dimensions=DIMS, walk_length=WL, num_walks=NW,
                                  window_size=WIN, workers=WK, seed=seed, beta=BETA,
                                  sim_kind=cfg["sim_kind"], gamma=cfg["gamma"]).fit(G_train)
    return normalize_embeddings(e)


def rank_metrics(scores_pos: np.ndarray, scores_neg: np.ndarray):
    """每个正样本与全部负样本比分数，返回 rank 数组（1 起）。"""
    # 名次 = 1 + 比它分数高的负样本数
    ranks = (scores_neg[None, :] >= scores_pos[:, None]).sum(axis=1) + 1
    return ranks


def main():
    print(f"🔬 [模块4] {DATASET.upper()}  Hits@K / MRR（协议 B）", flush=True)
    G = load_graph()
    path = os.path.join(OUT, f"{DATASET}_module4_rank.json")
    done = json.load(open(path)) if os.path.exists(path) else {}

    for seed in SEEDS:
        G_train, tp, _ = split_edges(G, 0.2, seed=seed)
        pos = list(G_train.edges())
        neg_train = sample_degree_matched(G_train, pos, seed=seed)
        tn = sample_degree_matched(G_train, tp, seed=seed + 1000)

        for mname, cfg in METHODS.items():
            if len(done.get(mname, [])) >= len(SEEDS):
                continue
            t0 = time.time()
            emb = build(cfg, G_train, seed)
            Xtr = np.vstack([get_edge_features(emb, pos, DIMS),
                             get_edge_features(emb, neg_train, DIMS)])
            ytr = np.array([1] * len(pos) + [0] * len(neg_train))
            clf = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr, ytr)
            sp = clf.predict_proba(get_edge_features(emb, tp, DIMS))[:, 1]
            sn = clf.predict_proba(get_edge_features(emb, tn, DIMS))[:, 1]
            ranks = rank_metrics(sp, sn)
            done.setdefault(mname, []).append({
                "hits10": float(np.mean(ranks <= 10)),
                "hits50": float(np.mean(ranks <= 50)),
                "mrr": float(np.mean(1.0 / ranks)),
                "median_rank": float(np.median(ranks)),
            })
            json.dump(done, open(path, "w"), ensure_ascii=False, indent=2)
            print(f"  seed={seed:2d} {mname:20s} H@10={done[mname][-1]['hits10']:.3f} "
                  f"H@50={done[mname][-1]['hits50']:.3f} MRR={done[mname][-1]['mrr']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n📊 {DATASET.upper()} 排序指标（mean ± std, n={len(SEEDS)}）", flush=True)
    for m, rows in done.items():
        for k in ("hits10", "hits50", "mrr"):
            a = np.array([r[k] for r in rows])
            print(f"  {m:22s} {k:8s} {a.mean():.4f} ± {a.std(ddof=1):.4f}", flush=True)
    print(f"💾 {path}", flush=True)


if __name__ == "__main__":
    main()
