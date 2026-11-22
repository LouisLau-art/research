"""
Week 12 混合预测器测试：结构相似度作为「显式特征」与嵌入拼接
══════════════════════════════════════════════════════════════════════
【这一页回答什么问题】

到目前为止，结构相似度只作用在「采样阶段」（改变游走方向），
A/B 两套协议下都没带来增益。那么换一个融合位置还有没有机会？

三个对照（同一批边划分、同一批负样本、同一个分类器）：
    E  仅嵌入特征      Hadamard(u,v)                      ← 表示学习基线
    H  仅结构特征      [CN, Jaccard, AA, RA]（各自归一化）  ← 启发式基线
    M  混合特征        concat(E, H)                        ← 本题要测的

在 A / B 两套协议下分别跑，看 M 能否显著超过 max(E, H)。

【关键设计】H 里不放任何「原始度数」类特征（如 deg(u)·deg(v)），
因为那正是上一轮查出来的度数捷径，会把结论污染。

运行:
  uv run python week12_hybrid_predictor.py cora
  uv run python week12_hybrid_predictor.py citeseer
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.graph_utils import all_non_edges, load_cora, load_citeseer, split_edges
from src.similarity import compute_scores
from src.embedding import DeepWalk
from src.advanced_walk import WeightedWalkEmbedding
from src.evaluation import normalize_embeddings, sample_degree_matched
from src.link_predictor import get_edge_features

OUT = "results/repeated"
SEEDS = list(range(1, 11))
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cora"
DIMS, WALK_LEN, NUM_WALKS, WINDOW, WORKERS = 64, 40, 10, 5, 1
HEURS = ("cn", "jaccard", "aa", "ra")


def structural_matrix(G, pairs):
    """把四种结构相似度拼成特征矩阵，每个指标先归一化到 [0,1] 量程。"""
    cols = []
    for m in HEURS:
        s = compute_scores(G, pairs, method=m).astype(float)
        mx = s.max()
        cols.append(s / mx if mx > 0 else s)
    return np.vstack(cols).T


def main():
    print(f"🔬 混合预测器: {DATASET.upper()}  seeds={SEEDS}", flush=True)
    G = load_cora() if DATASET == "cora" else load_citeseer()
    G = G[0] if isinstance(G, tuple) else G

    path = os.path.join(OUT, f"{DATASET}_hybrid.json")
    done = json.load(open(path)) if os.path.exists(path) else {}

    for emb_name, emb_fn in [
        ("DeepWalk", lambda Gt, s: DeepWalk(dimensions=DIMS, walk_length=WALK_LEN,
                                            num_walks=NUM_WALKS, window_size=WINDOW,
                                            workers=WORKERS, seed=s).fit(Gt)),
        ("Ours Jaccard γ=1", lambda Gt, s: WeightedWalkEmbedding(
            dimensions=DIMS, walk_length=WALK_LEN, num_walks=NUM_WALKS,
            window_size=WINDOW, workers=WORKERS, seed=s, beta=0.2,
            sim_kind="jaccard", gamma=1.0).fit(Gt)),
    ]:
        for proto in ("A", "B"):
            key = f"{emb_name}|{proto}"
            if len(done.get(key, {}).get("E", [])) >= len(SEEDS):
                continue
            for seed in SEEDS[len(done.get(key, {}).get("E", [])):]:
                t0 = time.time()
                G_train, tp, tn_rand = split_edges(G, 0.2, seed=seed)
                train_pos = list(G_train.edges())
                r = np.random.RandomState(seed)
                ne = all_non_edges(G_train)   # 确定性枚举，见 graph_utils.all_non_edges
                r.shuffle(ne)
                train_neg_rand = ne[:len(train_pos)]

                emb = emb_fn(G_train, seed)
                if proto == "B":
                    emb = normalize_embeddings(emb)
                    train_neg = sample_degree_matched(G_train, train_pos, seed=seed)
                    tn = sample_degree_matched(G_train, tp, seed=seed + 1000)
                else:
                    train_neg, tn = train_neg_rand, tn_rand

                # 边特征矩阵
                E_tr = np.vstack([get_edge_features(emb, train_pos, DIMS),
                                  get_edge_features(emb, train_neg, DIMS)])
                E_te = np.vstack([get_edge_features(emb, tp, DIMS),
                                  get_edge_features(emb, tn, DIMS)])
                H_tr = structural_matrix(G_train, train_pos + train_neg)
                H_te = structural_matrix(G_train, tp + tn)
                M_tr = np.hstack([E_tr, H_tr])
                M_te = np.hstack([E_te, H_te])

                y_tr = np.array([1] * len(train_pos) + [0] * len(train_neg))
                y_te = np.array([1] * len(tp) + [0] * len(tn))

                rec = done.setdefault(key, {"E": [], "H": [], "M": []})
                for tag, Xtr, Xte in (("E", E_tr, E_te), ("H", H_tr, H_te),
                                      ("M", M_tr, M_te)):
                    clf = LogisticRegression(max_iter=2000, random_state=42).fit(Xtr, y_tr)
                    rec[tag].append(float(roc_auc_score(y_te, clf.predict_proba(Xte)[:, 1])))
                done[key] = rec
                json.dump(done, open(path, "w"), ensure_ascii=False, indent=2)
                print(f"  {emb_name:18s}[{proto}] seed={seed:2d} "
                      f"E={rec['E'][-1]:.4f} H={rec['H'][-1]:.4f} M={rec['M'][-1]:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n📊 {DATASET.upper()} 混合预测器汇总 (10 划分均值)", flush=True)
    print(f"  {'配置':28s} {'E 仅嵌入':>10s} {'H 仅结构':>10s} {'M 混合':>10s}", flush=True)
    for key, rec in done.items():
        e, h, m = (np.array(rec[k]) for k in ("E", "H", "M"))
        print(f"  {key:28s} {e.mean():10.4f} {h.mean():10.4f} {m.mean():10.4f}", flush=True)


if __name__ == "__main__":
    main()
