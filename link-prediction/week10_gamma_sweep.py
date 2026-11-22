"""
Week 10 γ 锐化指数扫描 —— 验证「最优锐化程度随图密度变化」
══════════════════════════════════════════════════════════════════════
背景：Citeseer（平均度 2.8）上 Jaccard γ=4 取得 0.6669±0.0056，
      10 个划分全部胜过共同邻居 CN（0.6600±0.0045, 配对 p=0.0048）。
      而同样的 γ=4 在 Cora（平均度 3.9）上只有 0.7856，反而不如 γ=1 的 0.7988。

      这暗示「最优锐化程度与图的稀疏程度有关」：图越稀疏，越需要锐化。

本脚本补全 γ 的扫描范围：γ ∈ {0.5, 1, 2, 4, 8}，两个数据集各 10 个划分。

运行:
  uv run python week10_gamma_sweep.py cora
  uv run python week10_gamma_sweep.py citeseer
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx

from src.graph_utils import load_cora, load_citeseer, split_edges
from src.advanced_walk import WeightedWalkEmbedding
from src.link_predictor import LinkPredictor

OUT = "results/repeated"
SEEDS = list(range(1, 11))
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cora"

GAMMAS = [0.5, 1.0, 2.0, 4.0, 8.0]
BETA = 0.2


def build_neg(G_train, n, seed):
    ne = list(nx.non_edges(G_train))
    r = np.random.RandomState(seed); r.shuffle(ne)
    return ne[:n]


def main():
    print(f"🔬 γ 扫描: {DATASET.upper()}  γ={GAMMAS}  seeds={SEEDS}", flush=True)
    G = load_cora() if DATASET == "cora" else load_citeseer()
    G = G[0] if isinstance(G, tuple) else G

    path = os.path.join(OUT, f"{DATASET}_gamma_sweep.json")
    done = json.load(open(path)) if os.path.exists(path) else {}
    predictor = LinkPredictor(operator="hadamard")

    for g in GAMMAS:
        key = f"gamma={g}"
        aucs = done.get(key, [])
        if len(aucs) >= len(SEEDS):
            print(f"↩️  跳过 {key}（已完成）", flush=True)
            continue
        for seed in SEEDS[len(aucs):]:
            t0 = time.time()
            G_train, tp, tn = split_edges(G, 0.2, seed=seed)
            train_pos = list(G_train.edges())
            train_neg = build_neg(G_train, len(train_pos), seed)
            emb = WeightedWalkEmbedding(dimensions=64, walk_length=40, num_walks=10,
                                        sim_kind="jaccard", beta=BETA, gamma=g,
                                        window_size=5, workers=2, seed=seed)
            auc, _ = predictor.train_and_evaluate(emb.fit(G_train), 64,
                                                  train_pos, train_neg, tp, tn)
            aucs.append(auc)
            done[key] = aucs
            json.dump(done, open(path, "w"), ensure_ascii=False, indent=2)
            print(f"  γ={g:<4g} seed={seed:2d} AUC={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n📊 {DATASET.upper()} γ 扫描汇总", flush=True)
    for g in GAMMAS:
        a = np.array(done.get(f"gamma={g}", []))
        if len(a):
            print(f"  γ={g:<4g} {a.mean():.4f} ± {a.std(ddof=1):.4f}", flush=True)


if __name__ == "__main__":
    main()
