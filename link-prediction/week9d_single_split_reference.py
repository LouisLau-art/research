"""
week9d_single_split_reference.py —— 单次划分（seed=42）参照值
══════════════════════════════════════════════════════════════════════
中期答辩里有一页对比「单次划分的结论 vs 10 次划分的结论」。单次那一列原来是硬编码在
`build_figures.py` 里的，而那是可复现性修复之前跑的数字。本脚本在修复后的确定性配置下
重算一次，写入 results/repeated/single_split_seed42.json，供出图直接读取。

运行: PYTHONHASHSEED=0 .venv/bin/python week9d_single_split_reference.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.graph_utils import load_cora, load_citeseer, split_edges, all_non_edges
from src.embedding import DeepWalk, Node2Vec
from src.advanced_walk import WeightedWalkEmbedding
from src.link_predictor import LinkPredictor

DIMS, WL, NW, WIN, WORKERS = 64, 40, 10, 5, 1
SEED = 42
OUT = "results/repeated/single_split_seed42.json"

# 与中期答辩幻灯片「图 3」左半部分完全对应的三个方法
METHODS = {
    "DeepWalk": lambda G, s: DeepWalk(dimensions=DIMS, walk_length=WL, num_walks=NW,
                                      window_size=WIN, workers=WORKERS, seed=s).fit(G),
    "Node2Vec (DFS-like)": lambda G, s: Node2Vec(dimensions=DIMS, walk_length=WL, num_walks=NW,
                                                 p=2.0, q=0.5, window_size=WIN,
                                                 workers=WORKERS, seed=s).fit(G),
    "Ours (Weighted Walk)": lambda G, s: WeightedWalkEmbedding(
        dimensions=DIMS, walk_length=WL, num_walks=NW, window_size=WIN,
        workers=WORKERS, seed=s, beta=0.2, sim_kind="jaccard", gamma=1.0).fit(G),
}


def main():
    res = {}
    predictor = LinkPredictor(operator="hadamard")
    for ds, loader in (("cora", load_cora), ("citeseer", load_citeseer)):
        G = loader()
        G = G[0] if isinstance(G, tuple) else G
        G_train, tp, tn = split_edges(G, 0.2, seed=SEED)
        train_pos = list(G_train.edges())
        ne = all_non_edges(G_train)
        r = np.random.RandomState(SEED)
        r.shuffle(ne)
        train_neg = ne[:len(train_pos)]

        res[ds] = {}
        for name, fn in METHODS.items():
            auc, _ = predictor.train_and_evaluate(fn(G_train, SEED), DIMS,
                                                  train_pos, train_neg, tp, tn)
            res[ds][name] = round(float(auc), 4)
            print(f"  {ds:9s} {name:22s} AUC={auc:.4f}", flush=True)

    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"💾 {OUT}")


if __name__ == "__main__":
    main()