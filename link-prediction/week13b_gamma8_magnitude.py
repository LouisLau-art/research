"""
Week 13b —— 复核「γ=8 的虚假提升来自向量模长」
══════════════════════════════════════════════════════════════════════
同一份 γ=8 的嵌入，只改变「向量是否单位归一化」和「负样本怎么采」，
就能看出那部分 AUC 提升到底挂在结构信息上还是挂在模长上。

三个对照臂（共享同一份嵌入，不重复训练）：
    raw  + randomNeg        A 协议原样：原始向量 + 随机负样本
    norm + randomNeg        只消掉模长，负样本仍随机
    norm + degreeMatched    B 协议：模长与度数偏置都消掉
γ=1 作为参照一起跑。

运行: PYTHONHASHSEED=0 ./.venv/bin/python week13b_gamma8_magnitude.py [cora|citeseer]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.advanced_walk import WeightedWalkEmbedding
from src.evaluation import normalize_embeddings, sample_degree_matched
from src.graph_utils import all_non_edges, load_cora, load_citeseer, split_edges
from src.link_predictor import LinkPredictor

OUT, SEEDS = "results/repeated", list(range(1, 11))
DATASET = sys.argv[1] if len(sys.argv) > 1 else "citeseer"
DIMS, WL, NW, WIN, WK, BETA = 64, 40, 10, 5, 1, 0.2
GAMMAS = [1.0, 8.0]
ARMS = ["raw+randomNeg", "norm+randomNeg", "norm+degreeMatched"]


def main():
    G = (load_cora() if DATASET == "cora" else load_citeseer())[0]
    path = os.path.join(OUT, f"{DATASET}_gamma_magnitude.json")
    done = json.load(open(path)) if os.path.exists(path) else {}
    pred = LinkPredictor(operator="hadamard")

    for g in GAMMAS:
        for seed in SEEDS:
            keys = [f"g={g:g}|{a}" for a in ARMS]
            if all(len(done.get(k, {}).get("auc", [])) >= len(SEEDS) for k in keys):
                continue
            G_train, tp, tn_rand = split_edges(G, 0.2, seed=seed)
            pos = list(G_train.edges())
            r = np.random.RandomState(seed)
            ne = all_non_edges(G_train)
            r.shuffle(ne)
            neg_rand = ne[:len(pos)]
            tn_b = sample_degree_matched(G_train, tp, seed=seed + 1000)
            neg_b = sample_degree_matched(G_train, pos, seed=seed)

            t0 = time.time()
            emb_raw = WeightedWalkEmbedding(
                dimensions=DIMS, walk_length=WL, num_walks=NW, window_size=WIN,
                workers=WK, seed=seed, beta=BETA, sim_kind="jaccard", gamma=g).fit(G_train)
            emb_norm = normalize_embeddings(emb_raw)

            for arm, emb, tn, neg in (
                ("raw+randomNeg", emb_raw, tn_rand, neg_rand),
                ("norm+randomNeg", emb_norm, tn_rand, neg_rand),
                ("norm+degreeMatched", emb_norm, tn_b, neg_b),
            ):
                auc, ap = pred.train_and_evaluate(emb, DIMS, pos, neg, tp, tn)
                key = f"g={g:g}|{arm}"
                d = done.setdefault(key, {"auc": [], "ap": []})
                if len(d["auc"]) == seed - 1:
                    d["auc"].append(auc)
                    d["ap"].append(ap)
            json.dump(done, open(path, "w"), ensure_ascii=False, indent=2)
            print(f"  g={g:<4g} seed={seed:2d} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n📊 {DATASET.upper()} 模长捷径复核（mean ± std, n=10）", flush=True)
    for g in GAMMAS:
        for arm in ARMS:
            d = done.get(f"g={g:g}|{arm}", {"auc": [], "ap": []})
            a = np.array(d["auc"])
            if len(a):
                print(f"  g={g:<4g} {arm:20s} AUC {a.mean():.4f} ± {a.std(ddof=1):.4f}  "
                      f"AP {np.mean(d['ap']):.4f}", flush=True)
    print(f"💾 {path}", flush=True)


if __name__ == "__main__":
    main()
