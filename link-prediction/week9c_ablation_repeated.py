"""
Week 9c 消融实验（多种子版）
══════════════════════════════════════════════════════════════════════
week7_8 的消融只跑了单次划分（seed=42），结果不可靠（同配置不同种子的 AUC
标准差达 0.011，远大于消融格子之间的差异）。这里对两个关键超参
（嵌入维度 d、游走长度 l）各跑 5 个随机划分，报告 mean ± std。

运行: uv run python week9c_ablation_repeated.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx

from src.graph_utils import load_cora, split_edges
from src.advanced_walk import WeightedWalkEmbedding
from src.link_predictor import LinkPredictor

OUT = "results/repeated"
SEEDS = [1, 2, 3, 4, 5]
GRID = {
    "dims": [16, 32, 64, 128],
    "walk_len": [10, 20, 40, 80],
}


def build_neg(G_train, n, seed):
    ne = list(nx.non_edges(G_train))
    r = np.random.RandomState(seed); r.shuffle(ne)
    return ne[:n]


def main():
    G = load_cora()
    G = G[0] if isinstance(G, tuple) else G
    path = os.path.join(OUT, "cora_ablation.json")
    done = json.load(open(path)) if os.path.exists(path) else {}
    predictor = LinkPredictor(operator="hadamard")

    for axis, values in GRID.items():
        for v in values:
            key = f"{axis}={v}"
            aucs = done.get(key, [])
            if len(aucs) >= len(SEEDS):
                continue
            for seed in SEEDS[len(aucs):]:
                t0 = time.time()
                G_train, tp, tn = split_edges(G, 0.2, seed=seed)
                train_pos = list(G_train.edges())
                train_neg = build_neg(G_train, len(train_pos), seed)
                kw = dict(dimensions=64, walk_length=40)
                kw["dimensions" if axis == "dims" else "walk_length"] = v
                emb = WeightedWalkEmbedding(sim_kind="jaccard", beta=0.2, gamma=1.0,
                                            num_walks=10, window_size=5,
                                            workers=2, seed=seed, **kw)
                auc, _ = predictor.train_and_evaluate(emb.fit(G_train), kw["dimensions"],
                                                      train_pos, train_neg, tp, tn)
                aucs.append(auc)
                done[key] = aucs
                json.dump(done, open(path, "w"), ensure_ascii=False, indent=2)
                print(f"  {key:14s} seed={seed} AUC={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n📊 消融汇总 (Cora, 5 splits)", flush=True)
    for key in list(done):
        a = np.array(done[key])
        print(f"  {key:14s} {a.mean():.4f} ± {a.std(ddof=1):.4f}", flush=True)


if __name__ == "__main__":
    main()
