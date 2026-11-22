"""
build_m4_figures.py — CSIT998 模块 4 的三张图（重画版）
══════════════════════════════════════════════════════════════════════
重画原因：11-19 生成图 1/2/3 的临时脚本（/tmp/fig_m4.py、/tmp/fig_m4c.py）已丢失，
且 Cora 数据在修复可复现性后重跑过，旧图数字已过期。

本脚本读取 results/repeated/{cora,citeseer,facebook}_module4_ab.json，
输出 deliverables/csit998/ 下的三张矢量图（SVG）+ PNG 预览。

运行:
  cd link-prediction  # 从仓库根目录执行
  PYTHONHASHSEED=0 uv run python build_m4_figures.py
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "repeated")
OUT = os.path.join(os.path.dirname(HERE), "deliverables", "csit998")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#0f172a",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",   # 保留文本为文本，便于核对与二次编辑
})
INDIGO, AMBER, ROSE, GREEN, SLATE = "#4f46e5", "#d97706", "#e11d48", "#059669", "#94a3b8"
C_A, C_B = "#fda4af", "#a5b4fc"   # protocol A (red-ish) / B (blue-ish)

DATASETS = [("cora", "Cora"), ("citeseer", "Citeseer"), ("facebook", "Facebook")]

EMB_ORDER = ["DeepWalk", "Node2Vec (BFS-like)", "Node2Vec (DFS-like)",
             "Ours Jaccard g=1", "Ours RA g=1", "Ours AA g=1"]
EMB_SHORT = {"DeepWalk": "DeepWalk", "Node2Vec (BFS-like)": "N2V-BFS",
             "Node2Vec (DFS-like)": "N2V-DFS", "Ours Jaccard g=1": "Ours-J",
             "Ours RA g=1": "Ours-RA", "Ours AA g=1": "Ours-AA"}
HEUR_ORDER = ["degree-product", "CN", "Jaccard", "AA"]
HEUR_SHORT = {"degree-product": "deg. product", "CN": "CN",
              "Jaccard": "Jaccard", "AA": "AA"}


def load(ds):
    with open(os.path.join(RES, f"{ds}_module4_ab.json")) as f:
        return json.load(f)


def ms(v):
    a = np.array(v, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def n_seeds(d):
    """所有条目中最小的样本数 —— 数据完整性检查。"""
    return min(len(v["auc"]) for v in d.values())


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.svg"), format="svg",
                bbox_inches="tight", transparent=False)
    fig.savefig(os.path.join(OUT, f"{name}.png"), format="png", dpi=200,
                bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"  ✔ {name}")


# ══════════════════════════════════════════════════════════════════
# 图 1：协议漏洞 —— 纯度数基线在两套协议下的崩塌
# ══════════════════════════════════════════════════════════════════
def fig1_protocol_leak():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for ax, (ds, title) in zip(axes, DATASETS):
        d = load(ds)
        x = np.arange(len(HEUR_ORDER))
        a, ae, b, be = [], [], [], []
        for h in HEUR_ORDER:
            m, s = ms(d[f"{h}|A"]["auc"]); a.append(m); ae.append(s)
            m, s = ms(d[f"{h}|B"]["auc"]); b.append(m); be.append(s)

        ax.bar(x - 0.19, a, 0.36, yerr=ae, capsize=3, color=C_A,
               edgecolor=ROSE, linewidth=1.0, label="A · random negatives",
               error_kw=dict(ecolor="#9f1239", lw=0.9))
        ax.bar(x + 0.19, b, 0.36, yerr=be, capsize=3, color=C_B,
               edgecolor=INDIGO, linewidth=1.0, label="B · degree-matched",
               error_kw=dict(ecolor=INDIGO, lw=0.9))
        for xi, (av, bv) in enumerate(zip(a, b)):
            ax.text(xi - 0.19, av + 0.004, f"{av:.3f}", ha="center", fontsize=7.6,
                    color="#9f1239", rotation=90, va="bottom")
            ax.text(xi + 0.19, bv + 0.004, f"{bv:.3f}", ha="center", fontsize=7.6,
                    color="#312e81", rotation=90, va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([HEUR_SHORT[h] for h in HEUR_ORDER], fontsize=9,
                           rotation=18, ha="right")
        lo = min(min(a), min(b)) - 0.03
        ax.set_ylim(lo, max(max(a), max(b)) + 0.045)
        ax.axhline(0.5, ls=":", lw=1.1, color="#64748b", zorder=0)
        ax.set_title(title, fontsize=13, fontweight="bold", color="#0f172a", pad=8)
        if title == "Cora":
            ax.set_ylabel("AUC-ROC (10 splits)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("A one-number degree baseline is rewarded by random negatives, "
                 "and collapses to chance once negatives are degree-matched",
                 fontsize=11, color="#334155", y=1.02)
    fig.tight_layout()
    save(fig, "fig_m4_1_protocol_leak")


# ══════════════════════════════════════════════════════════════════
# 图 2：三数据集，两套协议下的完整排名
# ══════════════════════════════════════════════════════════════════
def fig2_ab_ranking():
    order = ["CN", "Jaccard", "AA"] + EMB_ORDER
    short = {**HEUR_SHORT, **EMB_SHORT}
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6))
    for ax, (ds, title) in zip(axes, DATASETS):
        d = load(ds)
        a, b = [], []
        for k in order:
            a.append(ms(d[f"{k}|A"]["auc"])[0])
            b.append(ms(d[f"{k}|B"]["auc"])[0])
        x = np.arange(len(order))
        ax.bar(x - 0.19, a, 0.36, color=C_A, edgecolor=ROSE, linewidth=0.9,
               label="A · random negatives")
        ax.bar(x + 0.19, b, 0.36, color=C_B, edgecolor=INDIGO, linewidth=0.9,
               label="B · degree-matched + unit-norm")
        for xi in x:
            ax.text(xi - 0.19, a[xi] + 0.003, f"{a[xi]:.3f}", ha="center",
                    fontsize=6.4, color="#9f1239", rotation=90, va="bottom")
            ax.text(xi + 0.19, b[xi] + 0.003, f"{b[xi]:.3f}", ha="center",
                    fontsize=6.4, color="#312e81", rotation=90, va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([short[k] for k in order], fontsize=8, rotation=38, ha="right")
        lo = min(min(a), min(b)) - 0.03
        ax.set_ylim(lo, max(max(a), max(b)) + 0.035)
        ax.axvline(2.5, ls=":", lw=1.0, color="#94a3b8", zorder=0)
        ax.set_title(title, fontsize=13, fontweight="bold", color="#0f172a", pad=8)
        ax.set_ylabel("AUC-ROC (10 splits)" if title == "Cora" else "")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Full ranking under the original vs the corrected protocol",
                 fontsize=11.5, color="#334155", y=1.01)
    fig.tight_layout()
    save(fig, "fig_m4_2_ab_ranking")


# ══════════════════════════════════════════════════════════════════
# 图 3：AUC 与 AP 并列 —— 证明排名不是 AUC 这个指标造成的假象
# ══════════════════════════════════════════════════════════════════
def fig3_auc_ap():
    order = ["CN", "Jaccard", "AA"] + EMB_ORDER
    short = {**HEUR_SHORT, **EMB_SHORT}
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 9.6), sharex="col")
    for r, (ds, title) in enumerate(DATASETS):
        d = load(ds)
        for c, metric in enumerate(("auc", "ap")):
            ax = axes[r][c]
            vals = [ms(d[f"{k}|B"][metric])[0] for k in order]
            errs = [ms(d[f"{k}|B"][metric])[1] for k in order]
            colors = [SLATE] * 3 + ["#818cf8"] * 3 + [INDIGO] * 3
            bars = ax.bar(np.arange(len(order)), vals, yerr=errs, capsize=3,
                          color=colors, edgecolor="white", linewidth=0.9,
                          error_kw=dict(ecolor="#475569", lw=0.9))
            for xi, (v, e) in enumerate(zip(vals, errs)):
                ax.text(xi, v + e + 0.002, f"{v:.3f}", ha="center", fontsize=7,
                        color="#0f172a", rotation=90, va="bottom")
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels([short[k] for k in order], fontsize=7.6,
                               rotation=38, ha="right")
            lo = min(v - e for v, e in zip(vals, errs))
            hi = max(v + e for v, e in zip(vals, errs))
            ax.set_ylim(lo - 0.03, hi + 0.04)
            ax.axvline(2.5, ls=":", lw=1.0, color="#94a3b8", zorder=0)
            metric_name = "AUC-ROC" if metric == "auc" else "Average Precision"
            ax.set_title(f"{title} · {metric_name}", fontsize=11.5,
                         fontweight="bold", color="#0f172a", pad=6)
            if c == 0:
                ax.set_ylabel("protocol B score")
    fig.suptitle("Under the corrected protocol the embeddings-vs-heuristics separation is the same for\n"
                 "AUC and AP, so the result is not an artefact of the AUC metric",
                 fontsize=11.5, color="#334155", y=0.995, linespacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save(fig, "fig_m4_3_auc_ap")


if __name__ == "__main__":
    print("🎨 重画模块 4 图表 ...")
    for ds, name in DATASETS:
        d = load(ds)
        print(f"  数据完整性 {name}: 每个条目 {n_seeds(d)} / 10 seeds")
    fig1_protocol_leak()
    fig2_ab_ranking()
    fig3_auc_ap()
    print("✅ 完成 →", OUT)