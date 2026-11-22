"""
Week 9 改进方案筛选实验
══════════════════════════════════════════════════════════════════════
背景：线性 Jaccard 加权（J+β，γ=1）相对 DeepWalk 的提升经 10 次配对检验
      不显著（Δ=+0.0044, p=0.44）。本脚本筛选若干改进方向：

  V1  Jaccard 锐化      γ=2
  V2  Jaccard 强锐化    γ=4
  V3  降低平滑项        β=0.05（γ=1）
  V4  Resource Allocation 加权（归一化后 γ=1）
  V5  Jaccard 加权 Node2Vec（叠加 DFS 偏置 p=2, q=0.5）← 开题公式原意
  V6  Adamic-Adar 加权（RA 的对数版本，归一化后 γ=1）

运行: uv run python week9b_improve_variants.py
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

VARIANTS = {
    "V1 Jaccard γ=2":        dict(sim_kind="jaccard", beta=0.2,  gamma=2.0),
    "V2 Jaccard γ=4":        dict(sim_kind="jaccard", beta=0.2,  gamma=4.0),
    "V3 Jaccard β=0.05":     dict(sim_kind="jaccard", beta=0.05, gamma=1.0),
    "V4 RA 加权":            dict(sim_kind="ra",      beta=0.2,  gamma=1.0),
    "V5 Jaccard+Node2Vec":   dict(sim_kind="jaccard", beta=0.2,  gamma=1.0, p=2.0, q=0.5),
    "V6 AA 加权":            dict(sim_kind="aa",      beta=0.2,  gamma=1.0),
}


def build_neg(G_train, n, seed):
    ne = list(nx.non_edges(G_train))
    r = np.random.RandomState(seed); r.shuffle(ne)
    return ne[:n]


def main():
    print(f"🔬 改进方案筛选: {DATASET.upper()}  (seeds={SEEDS})", flush=True)
    G = load_cora() if DATASET == "cora" else load_citeseer()
    G = G[0] if isinstance(G, tuple) else G

    path = os.path.join(OUT, f"{DATASET}_variants.json")
    done = json.load(open(path)) if os.path.exists(path) else {}

    predictor = LinkPredictor(operator="hadamard")

    for vname, cfg in VARIANTS.items():
        aucs = done.get(vname, [])
        if len(aucs) >= len(SEEDS):
            print(f"↩️  跳过 {vname}（已完成）", flush=True)
            continue
        for seed in SEEDS[len(aucs):]:
            t0 = time.time()
            G_train, tp, tn = split_edges(G, 0.2, seed=seed)
            train_pos = list(G_train.edges())
            train_neg = build_neg(G_train, len(train_pos), seed)
            emb = WeightedWalkEmbedding(dimensions=64, walk_length=40, num_walks=10,
                                        window_size=5, workers=2, seed=seed, **cfg)
            auc, _ = predictor.train_and_evaluate(emb.fit(G_train), 64,
                                                  train_pos, train_neg, tp, tn)
            aucs.append(auc)
            done[vname] = aucs
            json.dump(done, open(path, "w"), ensure_ascii=False, indent=2)
            print(f"  {vname:22s} seed={seed} AUC={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n📊 {DATASET.upper()} 改进方案汇总 (mean ± std, n={len(SEEDS)})", flush=True)
    for vname in VARIANTS:
        if vname not in done:
            continue
        a = np.array(done[vname])
        print(f"  {vname:22s} {a.mean():.4f} ± {a.std(ddof=1):.4f}", flush=True)


if __name__ == "__main__":
    main()
