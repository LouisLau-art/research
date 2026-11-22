"""
Week 9 重复实验：多种子均值±标准差 (填开题报告承诺的 10 次重复实验)
══════════════════════════════════════════════════════════════════════
运行: uv run python week9_repeated_experiments.py

目的：开题报告 Phase 2 明确承诺 "Each experiment will be repeated 10 times
with different random splits ... mean AUC along with standard deviation"。
此前的 week7_8 只跑了单次 (seed=42)，无法判断 +0.7% 的提升是真实效应还是随机波动。

本脚本输出：
  1. Cora / Citeseer 上 10 个随机划分的 mean ± std（各方法）
  2. Paired t-test 与 Wilcoxon 检验（Ours vs 各基线，逐划分配对）
  3. beta 敏感性 (5 个种子平均)

结果增量写入 results/repeated/*.json，随时可查看进度。
"""

import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx
from scipy import stats

from src.graph_utils import load_cora, load_citeseer, split_edges, all_non_edges
from src.similarity import compute_scores
from src.embedding import DeepWalk, Node2Vec, JaccardWeightedDeepWalk
from src.link_predictor import LinkPredictor
from sklearn.metrics import roc_auc_score

OUT_DIR = "results/repeated"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(1, 11))
DIMS, WALK_LEN, NUM_WALKS, WINDOW, WORKERS = 64, 40, 10, 5, 1
BETA = 0.2


def log(msg):
    print(msg, flush=True)


def dump(name, obj):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    log(f"💾 已写入 {path}")


def build_train_neg(G_train, n, seed):
    ne = all_non_edges(G_train)
    r = np.random.RandomState(seed)
    r.shuffle(ne)
    return ne[:n]


def run_dataset(dname):
    log("\n" + "=" * 70)
    log(f"🔬 数据集: {dname.upper()}   |   {len(SEEDS)} 个随机划分")
    log("=" * 70)

    G = load_cora() if dname == "cora" else load_citeseer()
    G = G[0] if isinstance(G, tuple) else G

    methods = ["Common Neighbors", "Jaccard Coefficient", "Adamic-Adar",
               "DeepWalk", "Node2Vec (BFS-like)", "Node2Vec (DFS-like)",
               "Ours (Weighted Walk)"]
    per_seed = {m: [] for m in methods}
    predictor = LinkPredictor(operator="hadamard")

    # --- 断点续跑：若已有部分结果则跳过已完成的种子 ---
    resume_path = os.path.join(OUT_DIR, f"{dname}_per_seed.json")
    done = 0
    if os.path.exists(resume_path):
        prev = json.load(open(resume_path))
        if all(len(prev.get(m, [])) == len(prev.get(methods[0], [])) for m in methods):
            n = len(prev.get(methods[0], []))
            if n > 0:
                for m in methods:
                    per_seed[m] = list(prev[m])
                done = n
                log(f"↩️  检测到已完成 {done} 个种子，从 seed={SEEDS[done] if done < len(SEEDS) else 'N/A'} 继续")

    for seed in SEEDS[done:]:
        t0 = time.time()
        G_train, test_pos, test_neg = split_edges(G, test_ratio=0.2, seed=seed)
        train_pos = list(G_train.edges())
        train_neg = build_train_neg(G_train, len(train_pos), seed)
        test_all = test_pos + test_neg
        y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))

        # --- 启发式（无需训练，但同样随划分变化） ---
        per_seed["Common Neighbors"].append(
            roc_auc_score(y_test, compute_scores(G_train, test_all, method="cn")))
        per_seed["Jaccard Coefficient"].append(
            roc_auc_score(y_test, compute_scores(G_train, test_all, method="jaccard")))
        per_seed["Adamic-Adar"].append(
            roc_auc_score(y_test, compute_scores(G_train, test_all, method="aa")))

        # --- DeepWalk ---
        dw = DeepWalk(dimensions=DIMS, walk_length=WALK_LEN, num_walks=NUM_WALKS,
                      window_size=WINDOW, workers=WORKERS, seed=seed)
        auc, _ = predictor.train_and_evaluate(dw.fit(G_train), DIMS, train_pos, train_neg, test_pos, test_neg)
        per_seed["DeepWalk"].append(auc)

        # --- Node2Vec BFS ---
        n2b = Node2Vec(dimensions=DIMS, walk_length=WALK_LEN, num_walks=NUM_WALKS,
                       p=0.5, q=2.0, window_size=WINDOW, workers=WORKERS, seed=seed)
        auc, _ = predictor.train_and_evaluate(n2b.fit(G_train), DIMS, train_pos, train_neg, test_pos, test_neg)
        per_seed["Node2Vec (BFS-like)"].append(auc)

        # --- Node2Vec DFS ---
        n2d = Node2Vec(dimensions=DIMS, walk_length=WALK_LEN, num_walks=NUM_WALKS,
                       p=2.0, q=0.5, window_size=WINDOW, workers=WORKERS, seed=seed)
        auc, _ = predictor.train_and_evaluate(n2d.fit(G_train), DIMS, train_pos, train_neg, test_pos, test_neg)
        per_seed["Node2Vec (DFS-like)"].append(auc)

        # --- Ours ---
        w = JaccardWeightedDeepWalk(dimensions=DIMS, walk_length=WALK_LEN, num_walks=NUM_WALKS,
                                    beta=BETA, window_size=WINDOW, workers=WORKERS, seed=seed)
        auc, _ = predictor.train_and_evaluate(w.fit(G_train), DIMS, train_pos, train_neg, test_pos, test_neg)
        per_seed["Ours (Weighted Walk)"].append(auc)

        log(f"  seed={seed:2d} 完成 ({time.time()-t0:.0f}s)  "
            f"DW={per_seed['DeepWalk'][-1]:.4f}  "
            f"DFS={per_seed['Node2Vec (DFS-like)'][-1]:.4f}  "
            f"Ours={per_seed['Ours (Weighted Walk)'][-1]:.4f}")
        dump(f"{dname}_per_seed.json", per_seed)

    # --- 汇总 ---
    summary = {}
    for m in methods:
        arr = np.array(per_seed[m])
        summary[m] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
                      "min": float(arr.min()), "max": float(arr.max())}

    # --- 配对显著性检验 (Ours vs 每个基线) ---
    ours = np.array(per_seed["Ours (Weighted Walk)"])
    sig = {}
    for m in methods:
        if m == "Ours (Weighted Walk)":
            continue
        base = np.array(per_seed[m])
        diff = ours - base
        t_p = float(stats.ttest_rel(ours, base).pvalue)
        try:
            w_p = float(stats.wilcoxon(ours, base).pvalue)
        except Exception:
            w_p = float("nan")
        sig[m] = {"mean_diff": float(diff.mean()),
                  "std_diff": float(diff.std(ddof=1)),
                  "t_pvalue": t_p,
                  "wilcoxon_pvalue": w_p,
                  "ours_better_in_n_splits": int((diff > 0).sum())}

    result = {"dataset": dname, "seeds": SEEDS, "per_seed": per_seed,
              "summary": summary, "significance": sig,
              "config": {"dims": DIMS, "walk_len": WALK_LEN, "num_walks": NUM_WALKS,
                         "window": WINDOW, "beta": BETA}}

    log("\n" + "-" * 70)
    log(f"📊 {dname.upper()} 汇总 ({len(SEEDS)} 次划分, mean ± std)")
    log("-" * 70)
    for m in methods:
        s = summary[m]
        log(f"  {m:26s} {s['mean']:.4f} ± {s['std']:.4f}   [{s['min']:.4f}, {s['max']:.4f}]")
    log("\n  Ours vs 基线（配对检验）:")
    for m, s in sig.items():
        log(f"    vs {m:26s} Δ={s['mean_diff']:+.4f}  t-p={s['t_pvalue']:.4f}  "
            f"wilcoxon-p={s['wilcoxon_pvalue']:.4f}  胜{s['ours_better_in_n_splits']}/{len(SEEDS)}")

    dump(f"{dname}_summary.json", result)
    return result


def run_beta_sensitivity(dname="cora", betas=(0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0)):
    log("\n" + "=" * 70)
    log(f"🎛️  beta 敏感性分析 ({dname.upper()}, 每档 5 个划分平均)")
    log("=" * 70)
    seeds = list(range(1, 6))
    G = load_cora() if dname == "cora" else load_citeseer()
    G = G[0] if isinstance(G, tuple) else G
    predictor = LinkPredictor(operator="hadamard")
    out = {}

    for beta in betas:
        aucs = []
        for seed in seeds:
            G_train, test_pos, test_neg = split_edges(G, test_ratio=0.2, seed=seed)
            train_pos = list(G_train.edges())
            train_neg = build_train_neg(G_train, len(train_pos), seed)
            w = JaccardWeightedDeepWalk(dimensions=DIMS, walk_length=WALK_LEN, num_walks=NUM_WALKS,
                                        beta=beta, window_size=WINDOW, workers=WORKERS, seed=seed)
            auc, _ = predictor.train_and_evaluate(w.fit(G_train), DIMS, train_pos, train_neg, test_pos, test_neg)
            aucs.append(auc)
        arr = np.array(aucs)
        out[str(beta)] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}
        log(f"  beta={beta:<7g} AUC={arr.mean():.4f} ± {arr.std(ddof=1):.4f}")
        dump(f"{dname}_beta_sensitivity.json", out)

    return out


if __name__ == "__main__":
    t_start = time.time()
    run_dataset("cora")
    run_dataset("citeseer")
    # run_beta_sensitivity("cora")  # 单独脚本运行，见 week9b
    log(f"\n✅ 全部完成，总耗时 {(time.time()-t_start)/60:.1f} 分钟")
