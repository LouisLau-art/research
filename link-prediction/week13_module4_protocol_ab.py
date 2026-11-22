"""
Week 13 / CSIT998 模块 4 —— 三数据集 × 双协议 × 双指标（AUC + AP）重复实验
══════════════════════════════════════════════════════════════════════
用途：给 CSIT998 小组初稿（模块 4：基于网络表示学习与节点结构相似度的
      社交网络链路预测）提供 Current Progress 一节的全部数字。

与已有脚本的关系：
  week11_corrected_protocol.py  只跑 Cora / Citeseer，只落 AUC
  week11b_heuristic_protocol.py 只跑无学习判边函数
  本脚本                        三数据集（含 Facebook）× 双协议 × AUC/AP，
                                并落盘 per-seed 明细供配对检验

协议定义（与 src/evaluation.py 一致）：
  A 原协议   随机负样本 + 原始向量
  B 修正协议 度数匹配负样本 + 单位归一化向量

运行:
  ./.venv/bin/python week13_module4_protocol_ab.py cora
  ./.venv/bin/python week13_module4_protocol_ab.py citeseer
  ./.venv/bin/python week13_module4_protocol_ab.py facebook
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx
from sklearn.metrics import average_precision_score, roc_auc_score

from src.advanced_walk import WeightedWalkEmbedding
from src.embedding import DeepWalk, Node2Vec
from src.evaluation import normalize_embeddings, sample_degree_matched
from src.graph_utils import all_non_edges, load_cora, load_citeseer, load_facebook, split_edges
from src.link_predictor import LinkPredictor, get_edge_features
from src.similarity import compute_scores

OUT = "results/repeated"
SEEDS = list(range(1, 11))
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cora"
DIMS, WALK_LEN, NUM_WALKS, WINDOW = 64, 40, 10, 5
# workers=1 且固定 PYTHONHASHSEED：gensim 多线程训练 + 未固定的哈希种子会让
# 同一划分、同一 seed 的 AUC 在 ±0.01 内抖动（实测 Cora σ=0.0005 但进程间可达 0.05），
# 会把方法间 0.004 量级的差距淹没。可复现性优先于训练速度。
WORKERS = 1
BETA = 0.2

# 嵌入类方法
EMB_METHODS = {
    "DeepWalk":            dict(kind="dw"),
    "Node2Vec (BFS-like)": dict(kind="n2v", p=0.5, q=2.0),
    "Node2Vec (DFS-like)": dict(kind="n2v", p=2.0, q=0.5),
    "Ours Jaccard g=1":    dict(kind="ww", sim_kind="jaccard", gamma=1.0),
    "Ours RA g=1":         dict(kind="ww", sim_kind="ra", gamma=1.0),
    "Ours AA g=1":         dict(kind="ww", sim_kind="aa", gamma=1.0),
}
# 无学习判边函数（秒级）
HEUR_METHODS = {"degree-product": None, "CN": "cn", "Jaccard": "jaccard", "AA": "aa"}


def load_graph():
    if DATASET == "cora":
        g = load_cora()
    elif DATASET == "citeseer":
        g = load_citeseer()
    else:
        g = load_facebook()
    return g[0] if isinstance(g, tuple) else g


def build_emb(cfg, G_train, seed):
    if cfg["kind"] == "dw":
        return DeepWalk(dimensions=DIMS, walk_length=WALK_LEN, num_walks=NUM_WALKS,
                        window_size=WINDOW, workers=WORKERS, seed=seed).fit(G_train)
    if cfg["kind"] == "n2v":
        return Node2Vec(dimensions=DIMS, walk_length=WALK_LEN, num_walks=NUM_WALKS,
                        p=cfg["p"], q=cfg["q"], window_size=WINDOW,
                        workers=WORKERS, seed=seed).fit(G_train)
    return WeightedWalkEmbedding(dimensions=DIMS, walk_length=WALK_LEN,
                                 num_walks=NUM_WALKS, window_size=WINDOW,
                                 workers=WORKERS, seed=seed, beta=BETA,
                                 sim_kind=cfg["sim_kind"], gamma=cfg["gamma"]).fit(G_train)


def auc_ap(emb, train_pos, train_neg, test_pos, test_neg):
    """Hadamard 边特征 + 逻辑回归；同时返回 AUC 与 AP（用同一组预测概率）。"""
    pred = LinkPredictor(operator="hadamard")
    xtr = np.vstack([
        get_edge_features(emb, train_pos, DIMS, "hadamard"),
        get_edge_features(emb, train_neg, DIMS, "hadamard"),
    ])
    ytr = np.array([1] * len(train_pos) + [0] * len(train_neg))
    xte = np.vstack([
        get_edge_features(emb, test_pos, DIMS, "hadamard"),
        get_edge_features(emb, test_neg, DIMS, "hadamard"),
    ])
    yte = np.array([1] * len(test_pos) + [0] * len(test_neg))
    pred.clf.fit(xtr, ytr)
    p = pred.clf.predict_proba(xte)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def main():
    print(f"🔬 [模块4] {DATASET.upper()}  协议 A/B × AUC/AP  seeds={SEEDS}", flush=True)
    G = load_graph()
    path = os.path.join(OUT, f"{DATASET}_module4_ab.json")
    done = json.load(open(path)) if os.path.exists(path) else {}

    for seed in SEEDS:
        G_train, tp, tn_rand = split_edges(G, 0.2, seed=seed)
        train_pos = list(G_train.edges())
        r = np.random.RandomState(seed)
        ne = all_non_edges(G_train)   # 确定性枚举，见 graph_utils.all_non_edges
        r.shuffle(ne)
        train_neg_rand = ne[:len(train_pos)]
        tn_b = sample_degree_matched(G_train, tp, seed=seed + 1000)
        train_neg_b = sample_degree_matched(G_train, train_pos, seed=seed)

        # ── 嵌入类方法 ──────────────────────────────────────────
        for mname, cfg in EMB_METHODS.items():
            if (len(done.get(f"{mname}|A", {}).get("auc", [])) >= len(SEEDS)
                    and len(done.get(f"{mname}|B", {}).get("auc", [])) >= len(SEEDS)):
                continue
            t0 = time.time()
            emb_raw = build_emb(cfg, G_train, seed)
            emb_norm = normalize_embeddings(emb_raw)

            a_auc, a_ap = auc_ap(emb_raw, train_pos, train_neg_rand, tp, tn_rand)
            b_auc, b_ap = auc_ap(emb_norm, train_pos, train_neg_b, tp, tn_b)
            for proto, auc, ap in (("A", a_auc, a_ap), ("B", b_auc, b_ap)):
                k = f"{mname}|{proto}"
                d = done.setdefault(k, {"auc": [], "ap": []})
                if len(d["auc"]) == seed - 1:
                    d["auc"].append(auc)
                    d["ap"].append(ap)
            json.dump(done, open(path, "w"), ensure_ascii=False, indent=2)
            print(f"  seed={seed:2d} {mname:20s} A {a_auc:.4f}/{a_ap:.4f}  "
                  f"B {b_auc:.4f}/{b_ap:.4f}  ({time.time()-t0:.0f}s)", flush=True)

        # ── 无学习判边函数 ──────────────────────────────────────
        deg = dict(G_train.degree())
        for hname, skey in HEUR_METHODS.items():
            for proto, tn in (("A", tn_rand), ("B", tn_b)):
                y = np.array([1] * len(tp) + [0] * len(tn))
                pairs = tp + tn
                if hname == "degree-product":
                    s = np.array([deg[u] * deg[v] for u, v in pairs], float)
                else:
                    s = compute_scores(G_train, pairs, method=skey)
                k = f"{hname}|{proto}"
                d = done.setdefault(k, {"auc": [], "ap": []})
                if len(d["auc"]) == seed - 1:
                    d["auc"].append(roc_auc_score(y, s))
                    d["ap"].append(average_precision_score(y, s))
        json.dump(done, open(path, "w"), ensure_ascii=False, indent=2)
        print(f"  seed={seed:2d} heuristics done", flush=True)

    print(f"\n📊 {DATASET.upper()} 汇总（mean ± std, n=10）", flush=True)
    print(f"  {'方法':22s} {'A-AUC':>16s} {'A-AP':>16s} {'B-AUC':>16s} {'B-AP':>16s}", flush=True)
    for mname in list(EMB_METHODS) + list(HEUR_METHODS):
        row = []
        for proto in ("A", "B"):
            d = done.get(f"{mname}|{proto}", {"auc": [], "ap": []})
            for arr in (d["auc"], d["ap"]):
                a = np.array(arr)
                row.append(f"{a.mean():.4f}±{a.std(ddof=1):.4f}" if len(a) else "—")
        print(f"  {mname:22s} {row[0]:>16s} {row[1]:>16s} {row[2]:>16s} {row[3]:>16s}", flush=True)
    print(f"💾 {path}", flush=True)


if __name__ == "__main__":
    main()
