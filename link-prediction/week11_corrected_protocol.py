"""
Week 11 修正评估协议对照实验
══════════════════════════════════════════════════════════════════════
问题：Week 9-10 用的评估协议（随机负采样 + 原始向量 Hadamard → 逻辑回归）存在
      两个会让「与方法无关的信号」混进 AUC 的漏洞：

  1. 向量模长捷径：Hadamard 特征幅值随节点度数增长，分类器可以用「模长大小」判边
  2. 随机负样本低估度数：真实边天然连接高度节点，随机负样本则不然
     （Citeseer 正边度积 27.0 vs 负边 5.3；Cora 42.3 vs 9.0）

本脚本在同一批边划分上，对每种方法跑两套协议：

  A 协议（原）：随机负样本 + 原始向量
  B 协议（修正）：度数匹配负样本 + 单位归一化向量

比较 A / B 下各方法 AUC 与排名是否变化。

运行:
  uv run python week11_corrected_protocol.py cora
  uv run python week11_corrected_protocol.py citeseer
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx

from src.graph_utils import all_non_edges, load_cora, load_citeseer, split_edges
from src.embedding import DeepWalk, Node2Vec
from src.advanced_walk import WeightedWalkEmbedding
from src.evaluation import normalize_embeddings, sample_degree_matched
from src.link_predictor import LinkPredictor

OUT = "results/repeated"
SEEDS = list(range(1, 11))
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cora"
DIMS, WALK_LEN, NUM_WALKS, WINDOW, WORKERS = 64, 40, 10, 5, 1
BETA = 0.2

METHODS = {
    "DeepWalk":                dict(kind="dw"),
    "Node2Vec (DFS-like)":     dict(kind="n2v", p=2.0, q=0.5),
    "Ours Jaccard γ=1":        dict(kind="ww", sim_kind="jaccard", gamma=1.0),
    "Ours RA γ=1":             dict(kind="ww", sim_kind="ra", gamma=1.0),
    "Ours AA γ=1":             dict(kind="ww", sim_kind="aa", gamma=1.0),
    "Ours Jaccard γ=4":        dict(kind="ww", sim_kind="jaccard", gamma=4.0),
}


def build_emb(cfg, G_train, seed):
    if cfg["kind"] == "dw":
        return DeepWalk(dimensions=DIMS, walk_length=WALK_LEN, num_walks=NUM_WALKS,
                        window_size=WINDOW, workers=WORKERS, seed=seed).fit(G_train)
    if cfg["kind"] == "n2v":
        return Node2Vec(dimensions=DIMS, walk_length=WALK_LEN, num_walks=NUM_WALKS,
                        p=cfg["p"], q=cfg["q"], window_size=WINDOW,
                        workers=WORKERS, seed=seed).fit(G_train)
    if cfg["kind"] == "ww":
        return WeightedWalkEmbedding(dimensions=DIMS, walk_length=WALK_LEN,
                                     num_walks=NUM_WALKS, window_size=WINDOW,
                                     workers=WORKERS, seed=seed, beta=BETA,
                                     sim_kind=cfg["sim_kind"], gamma=cfg["gamma"]).fit(G_train)


def main():
    print(f"🔬 修正评估协议: {DATASET.upper()}  seeds={SEEDS}", flush=True)
    G = load_cora() if DATASET == "cora" else load_citeseer()
    G = G[0] if isinstance(G, tuple) else G

    path = os.path.join(OUT, f"{DATASET}_protocol_ab.json")
    done = json.load(open(path)) if os.path.exists(path) else {}
    pred = LinkPredictor(operator="hadamard")

    for mname, cfg in METHODS.items():
        for proto in ("A", "B"):
            key = f"{mname}|{proto}"
            aucs = done.get(key, [])
            if len(aucs) >= len(SEEDS):
                continue
            for seed in SEEDS[len(aucs):]:
                t0 = time.time()
                G_train, tp, tn_rand = split_edges(G, 0.2, seed=seed)
                train_pos = list(G_train.edges())
                r = np.random.RandomState(seed)
                ne = all_non_edges(G_train)   # 确定性枚举，见 graph_utils.all_non_edges
                r.shuffle(ne)
                train_neg_rand = ne[:len(train_pos)]

                if proto == "A":
                    emb, train_neg, tn = build_emb(cfg, G_train, seed), train_neg_rand, tn_rand
                else:
                    emb = normalize_embeddings(build_emb(cfg, G_train, seed))
                    train_neg = sample_degree_matched(G_train, train_pos, seed=seed)
                    tn = sample_degree_matched(G_train, tp, seed=seed + 1000)

                auc, _ = pred.train_and_evaluate(emb, DIMS, train_pos, train_neg, tp, tn)
                aucs.append(auc)
                done[key] = aucs
                json.dump(done, open(path, "w"), ensure_ascii=False, indent=2)
                print(f"  {mname:20s} [{proto}] seed={seed:2d} AUC={auc:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n📊 {DATASET.upper()} A/B 协议对照", flush=True)
    print(f"  {'方法':22s} {'A 原协议':>18s} {'B 修正协议':>18s}", flush=True)
    for mname in METHODS:
        a = np.array(done.get(f"{mname}|A", []))
        b = np.array(done.get(f"{mname}|B", []))
        if len(a) and len(b):
            print(f"  {mname:22s} {a.mean():.4f}±{a.std(ddof=1):.4f}   "
                  f"{b.mean():.4f}±{b.std(ddof=1):.4f}", flush=True)


if __name__ == "__main__":
    main()
