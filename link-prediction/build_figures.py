"""
build_figures.py — 中期答辩图表生成（矢量 SVG）
══════════════════════════════════════════════════════════════════════
数据来源全部是 results/repeated/*.json（10 次随机划分的重复实验结果）。

输出: deliverables/figures/*.svg
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "repeated")
OUT = os.path.join(os.path.dirname(HERE), "deliverables", "figures")
os.makedirs(OUT, exist_ok=True)

# ── 统一视觉规范 ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
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
})
INDIGO, AMBER, SLATE, GREEN, ROSE = "#4f46e5", "#d97706", "#94a3b8", "#059669", "#e11d48"


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"  ✔ {name}")


def ms(v):
    a = np.array(v)
    return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0


# ══════════════════════════════════════════════════════════════════
# 图 1：主结果（Cora / Citeseer），7 种方法，误差棒 = 10 次划分的 std
# ══════════════════════════════════════════════════════════════════
def fig_main():
    cora = load("cora_summary.json")["summary"]
    cite = load("citeseer_summary.json")["summary"]

    order = [
        ("Common Neighbors", "CN"),
        ("Jaccard Coefficient", "JC"),
        ("Adamic-Adar", "AA"),
        ("DeepWalk", "DeepWalk"),
        ("Node2Vec (BFS-like)", "N2V-BFS"),
        ("Node2Vec (DFS-like)", "N2V-DFS"),
        ("Ours (Weighted Walk)", "Ours"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    for ax, data, title in zip(axes, [cora, cite], ["Cora", "Citeseer"]):
        names = [short for _, short in order]
        means = [data[k]["mean"] for k, _ in order]
        stds = [data[k]["std"] for k, _ in order]
        colors = ["#cbd5e1"] * 3 + ["#94a3b8"] * 3 + [INDIGO]
        bars = ax.bar(names, means, yerr=stds, capsize=4, color=colors,
                      edgecolor="white", linewidth=1.0, width=0.72,
                      error_kw=dict(ecolor="#334155", lw=1.1, capthick=1.1))
        bars[-1].set_color(INDIGO)
        bars[-1].set_edgecolor(INDIGO)

        # 启发式上限参考线（CN）
        cn = data["Common Neighbors"]["mean"]
        ax.axhline(cn, ls="--", lw=1.2, color=AMBER, alpha=0.85, zorder=0)

        ax.set_title(f"{title}", fontsize=15, fontweight="bold", color="#0f172a", pad=24)
        ax.text(0.5, 1.012, f"best simple heuristic (CN) = {cn:.3f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9.5, color="#b45309")
        ax.set_ylabel("AUC-ROC (mean of 10 splits)" if title == "Cora" else "")
        lo = min(means) - max(stds) - 0.03
        hi = max(means) + max(stds) + 0.035
        ax.set_ylim(min(lo, cn - 0.03), hi)
        ax.tick_params(axis="x", labelrotation=28, labelsize=10)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")

    fig.suptitle("Link-prediction AUC on two benchmark graphs  (error bar = std over 10 random edge splits)",
                 fontsize=12.5, color="#334155", y=1.03)
    fig.tight_layout()
    save(fig, "fig1_main_results.svg")


# ══════════════════════════════════════════════════════════════════
# 图 2：逐划分波动（为什么必须做重复实验）
# ══════════════════════════════════════════════════════════════════
def fig_variance():
    per = load("cora_per_seed.json")
    sel = [("DeepWalk", SLATE), ("Node2Vec (DFS-like)", AMBER), ("Ours (Weighted Walk)", INDIGO)]

    fig, ax = plt.subplots(figsize=(11.6, 4.3))
    xs = np.arange(1, 11)
    for name, color in sel:
        y = np.array(per[name])
        ax.plot(xs, y, "-o", ms=6, lw=1.8, color=color, label=name,
                mfc="white", mew=1.6, zorder=3)
        ax.axhline(y.mean(), ls=":", lw=1.4, color=color, alpha=0.55, zorder=1)

    dw = np.array(per["DeepWalk"]); o = np.array(per["Ours (Weighted Walk)"])
    ax.text(0.985, 0.045,
            f"mean gap  DeepWalk → Ours = {o.mean()-dw.mean():+.4f}\n"
            f"spread of a single method  σ ≈ {o.std(ddof=1):.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1", lw=0.9))

    ax.set_xlabel("random edge split #")
    ax.set_ylabel("AUC-ROC")
    ax.set_xticks(xs)
    ax.set_xlim(0.4, 10.6)
    ax.set_title("Same models, different random split → AUC swings more than the method gap (Cora)",
                 fontsize=12.5, color="#0f172a", pad=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_seed_variance.svg")


# ══════════════════════════════════════════════════════════════════
# 图 3：单次(seed=42) vs 10 次重复 —— 单次结果有多误导
# ══════════════════════════════════════════════════════════════════
def fig_single_vs_repeated():
    ref = load("single_split_seed42.json")
    single = {"Cora": ref["cora"], "Citeseer": ref["citeseer"]}
    repeated = {"Cora": load("cora_summary.json")["summary"],
                "Citeseer": load("citeseer_summary.json")["summary"]}
    order = [("DeepWalk", SLATE), ("Node2Vec (DFS-like)", AMBER), ("Ours (Weighted Walk)", INDIGO)]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1))
    for ax, ds in zip(axes, ["Cora", "Citeseer"]):
        labels = [k for k, _ in order]
        s = [single[ds][k] for k, _ in order]
        r = [repeated[ds][k]["mean"] for k, _ in order]
        e = [repeated[ds][k]["std"] for k, _ in order]
        x = np.arange(len(labels))
        ax.bar(x - 0.19, s, 0.36, label="single split (seed=42)", color="#e2e8f0",
               edgecolor="#94a3b8", linewidth=1.0)
        ax.bar(x + 0.19, r, 0.36, yerr=e, capsize=3.5, label="10 splits (mean±std)",
               color="#c7d2fe", edgecolor=INDIGO, linewidth=1.2,
               error_kw=dict(ecolor=INDIGO, lw=1.1, capthick=1.1))
        for xi, (sv, rv) in enumerate(zip(s, r)):
            ax.text(xi - 0.19, sv + 0.002, f"{sv:.4f}", ha="center", fontsize=8.6, color="#475569")
            ax.text(xi + 0.19, rv + e[xi] + 0.002, f"{rv:.4f}", ha="center", fontsize=8.6,
                    color="#3730a3", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(ds, fontsize=13.5, fontweight="bold", color="#0f172a", pad=10)
        ax.set_ylim(min(min(s), min(r)) - 0.035, max(max(s), max(r)) + 0.03)
        if ds == "Cora":
            ax.set_ylabel("AUC-ROC")
    axes[0].legend(frameon=False, fontsize=9.5, loc="lower left")
    fig.suptitle("A single split can reverse the ranking — repeated evaluation corrects it",
                 fontsize=12.5, color="#334155", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_single_vs_repeated.svg")


# ══════════════════════════════════════════════════════════════════
# 图 4：改进方案筛选
# ══════════════════════════════════════════════════════════════════
def fig_variants():
    base = load("cora_per_seed.json")
    var = load("cora_variants.json")

    items = [
        ("DeepWalk", *ms(base["DeepWalk"]), SLATE),
        ("N2V-DFS", *ms(base["Node2Vec (DFS-like)"]), AMBER),
        ("Ours\nJaccard γ=1", *ms(base["Ours (Weighted Walk)"]), "#818cf8"),
        ("V1\nγ=2", *ms(var["V1 Jaccard γ=2"]), SLATE),
        ("V2\nγ=4", *ms(var["V2 Jaccard γ=4"]), SLATE),
        ("V3\nβ=0.05", *ms(var["V3 Jaccard β=0.05"]), SLATE),
        ("V4\nRA weight", *ms(var["V4 RA 加权"]), GREEN),
        ("V5\nJac+N2V", *ms(var["V5 Jaccard+Node2Vec"]), GREEN),
    ]
    labels = [i[0] for i in items]
    means = [i[1] for i in items]
    stds = [i[2] for i in items]
    colors = [i[3] for i in items]

    fig, ax = plt.subplots(figsize=(11.8, 4.6))
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="white", linewidth=1.0, width=0.68,
                  error_kw=dict(ecolor="#334155", lw=1.1, capthick=1.1))
    ax.axhline(means[1], ls="--", lw=1.3, color=AMBER, alpha=0.85, zorder=0)
    ax.axhline(means[0], ls=":", lw=1.3, color=SLATE, zorder=0)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.0018, f"{m:.4f}",
                ha="center", fontsize=9.6, fontweight="bold", color="#0f172a")
    ax.set_ylabel("AUC-ROC (10 splits)")
    lo = min(means) - max(stds) - 0.010
    ax.set_ylim(lo, max(m + s for m, s in zip(means, stds)) + 0.011)
    ax.set_title("Screening five redesigns of the similarity-weighted walk (Cora)",
                 fontsize=12.5, color="#0f172a", pad=12)
    fig.text(0.5, -0.035,
             "sharpening (V1–V3) drops below DeepWalk  ·  only the degree-aware RA weight (V4) holds on par with the strongest baseline",
             ha="center", fontsize=9.5, color="#475569", style="italic")
    fig.tight_layout()
    save(fig, "fig4_variants.svg")


# ══════════════════════════════════════════════════════════════════
# 图 5：消融（维度 / 游走长度）
# ══════════════════════════════════════════════════════════════════
def fig_ablation():
    d = load("cora_ablation.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))

    for ax, axis, xs, title, xlabel in [
        (axes[0], "dims", [16, 32, 64, 128], "Embedding dimension", "dimension d"),
        (axes[1], "walk_len", [10, 20, 40, 80], "Walk length", "walk length l"),
    ]:
        ys, es = [], []
        for v in xs:
            key = f"{axis}={v}"
            if key in d:
                m, s = ms(d[key]); ys.append(m); es.append(s)
            else:
                ys.append(np.nan); es.append(0)
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=7, lw=2, color=INDIGO,
                    mfc="white", mew=2, capsize=4, ecolor="#6366f1")
        ax.set_xticks(xs)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=12.5, fontweight="bold", color="#0f172a", pad=10)
        if axis == "dims":
            ax.set_ylabel("AUC-ROC (5 splits)")
    fig.suptitle("Hyper-parameter ablation of the weighted walk (Cora)",
                 fontsize=12, color="#334155", y=1.03)
    fig.tight_layout()
    save(fig, "fig5_ablation.svg")


# ══════════════════════════════════════════════════════════════════
# 图 6：进度甘特（对照开题计划 vs 实际）
# ══════════════════════════════════════════════════════════════════
def fig_gantt():
    phases = [
        ("P1 Literature review · setup", 0, 2, "done"),
        ("P2 Baselines · 3 datasets", 2, 4, "done"),
        ("P3 Proposed method · tuning", 4, 6, "active"),
        ("P4 Analysis · thesis writing", 6, 8, "todo"),
        ("P5 First draft · review", 8, 9, "todo"),
        ("P6 Final version · defense", 9, 10, "todo"),
    ]
    colors = {"done": "#4f46e5", "active": "#d97706", "todo": "#e2e8f0"}
    edge = {"done": "#4f46e5", "active": "#d97706", "todo": "#cbd5e1"}

    fig, ax = plt.subplots(figsize=(11.4, 3.5))
    for i, (label, start, end, state) in enumerate(phases):
        y = len(phases) - 1 - i
        ax.barh(y, end - start, left=start, height=0.52, color=colors[state],
                edgecolor=edge[state], linewidth=1.4, zorder=3)
        ax.text(-0.15, y, label, ha="right", va="center", fontsize=10.5,
                color="#0f172a")
    # 今天的位置（Phase 3 进行中，第 4.1 个月）
    ax.axvline(4.13, ls="--", lw=1.8, color=ROSE, zorder=4)
    ax.text(4.2, len(phases) - 0.55, "today\n(4.1 / 10 months)", fontsize=9.5, color=ROSE,
            fontweight="bold", va="top")

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, len(phases) - 0.35)
    ax.set_yticks([])
    ax.set_xticks(range(0, 11))
    ax.set_xticklabels(["Jul'26", "Aug", "Sep", "Oct", "Nov", "Dec",
                        "Jan'27", "Feb", "Mar", "Apr", "–"], fontsize=9.5)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_title("Research schedule: on track against the proposal timeline",
                 fontsize=12.5, color="#0f172a", pad=12, loc="left")
    fig.tight_layout()
    save(fig, "fig6_gantt.svg")


# ══════════════════════════════════════════════════════════════════
# 图 7：加权游走示意（一个小图的 Jaccard 权重示意，矢量化）
# ══════════════════════════════════════════════════════════════════
def fig_walk_schematic():
    """
    画一个 3 邻居的转移概率示意：J(v,a)=0.17 / J(v,b)=0.40 / J(v,c)=0.14，
    在 beta=0.2 下得到明显不同的转移概率，直观说明「为什么加权会改变游走方向」。
    """
    import networkx as nx

    G = nx.Graph()
    edges = [("v", "a"), ("v", "b"), ("v", "c"),
             ("a", "b"), ("b", "c"),
             ("a", "u1"), ("a", "u2"), ("b", "u3"),
             ("c", "u4"), ("c", "u5"), ("c", "u6")]
    G.add_edges_from(edges)

    nv = set(G.neighbors("v"))
    jac = {}
    for x in "abc":
        nx_ = set(G.neighbors(x))
        jac[x] = len(nv & nx_) / len(nv | nx_)

    beta = 0.2
    tot = sum(jac[x] + beta for x in "abc")
    prob = {x: (jac[x] + beta) / tot for x in "abc"}

    pos = {"v": (0.0, 0.0),
           "a": (-1.85, 1.05), "b": (0.0, 1.45), "c": (1.85, 1.05),
           "u1": (-2.95, 1.95), "u2": (-1.35, 2.45), "u3": (0.0, 2.55),
           "u4": (1.35, 2.45), "u5": (2.95, 1.95), "u6": (2.15, 2.6)}

    fig, ax = plt.subplots(figsize=(8.2, 6.0))

    # 背景的 2 跳边（浅灰）
    for u, w in G.edges():
        if "v" not in (u, w):
            ax.plot([pos[u][0], pos[w][0]], [pos[u][1], pos[w][1]],
                    color="#e2e8f0", lw=1.6, zorder=1, solid_capstyle="round")

    # v 出发的三条边：宽度与深浅随转移概率变化
    for x in "abc":
        p = prob[x]
        ax.plot([pos["v"][0], pos[x][0]], [pos["v"][1], pos[x][1]],
                color=INDIGO, lw=2.0 + 11 * p, alpha=0.22 + 0.72 * p, zorder=2,
                solid_capstyle="round")
        # 标注放在边的中点偏外侧
        lx, ly = {"a": (-1.28, 0.46), "b": (0.72, 0.78), "c": (1.28, 0.46)}[x]
        ax.text(lx, ly, f"J = {jac[x]:.2f}\np = {p:.2f}",
                fontsize=9.2, color="#1e3a8a", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#c7d2fe", lw=0.9))

    for n_, (x, y) in pos.items():
        if n_ == "v":
            c, s_, tc = INDIGO, 1150, "white"
        elif n_ in "abc":
            c, s_, tc = "#f59e0b", 780, "white"
        else:
            c, s_, tc = "#e2e8f0", 430, "#64748b"
        ax.scatter([x], [y], s=s_, color=c, edgecolor="white", linewidth=2.2, zorder=4)
        ax.text(x, y, n_, ha="center", va="center", fontsize=11.5, fontweight="bold",
                color=tc, zorder=5)

    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-0.75, 2.95)
    ax.axis("off")
    ax.set_title("Transition weight  P(v→x) ∝ ( J(v,x) + β ) / Σ", fontsize=13.5,
                 color="#0f172a", pad=2)
    fig.tight_layout()
    save(fig, "fig7_walk_schematic.svg")

# ══════════════════════════════════════════════════════════════════
# 图 8：评估协议漏洞 —— 随机负采样白送的 AUC
# ══════════════════════════════════════════════════════════════════
def fig_protocol_leak():
    cora = load("cora_heuristics_ab.json")
    cite = load("citeseer_heuristics_ab.json")
    items = [("degree-product", "degree product\n(no structure)"),
             ("CN", "CN"), ("Jaccard", "Jaccard"), ("AA", "AA")]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for ax, data, title in zip(axes, [cora, cite], ["Cora", "Citeseer"]):
        x = np.arange(len(items))
        a = [float(np.mean(data[n]["A"])) for n, _ in items]
        ae = [float(np.std(data[n]["A"], ddof=1)) for n, _ in items]
        b = [float(np.mean(data[n]["B"])) for n, _ in items]
        be = [float(np.std(data[n]["B"], ddof=1)) for n, _ in items]
        ax.bar(x - 0.19, a, 0.36, yerr=ae, capsize=3.5, color="#fda4af",
               edgecolor="#e11d48", linewidth=1.1,
               label="A · random negatives",
               error_kw=dict(ecolor="#9f1239", lw=1.0))
        ax.bar(x + 0.19, b, 0.36, yerr=be, capsize=3.5, color="#a5b4fc",
               edgecolor=INDIGO, linewidth=1.1,
               label="B · degree-matched negatives",
               error_kw=dict(ecolor=INDIGO, lw=1.0))
        for xi, (av, bv) in enumerate(zip(a, b)):
            ax.text(xi - 0.19, av + ae[xi] + 0.004, f"{av:.3f}", ha="center",
                    fontsize=8.8, color="#9f1239", fontweight="bold")
            ax.text(xi + 0.19, bv + be[xi] + 0.004, f"{bv:.3f}", ha="center",
                    fontsize=8.8, color="#312e81", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in items], fontsize=9.5)
        ax.set_ylim(0.46, 0.80)
        ax.axhline(0.5, ls=":", lw=1.2, color="#64748b", zorder=0)
        ax.set_title(title, fontsize=14, fontweight="bold", color="#0f172a", pad=10)
        if title == "Cora":
            ax.set_ylabel("AUC-ROC")
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.suptitle("Random negative sampling awards AUC to pure degree\n"
                 "a one-number degree baseline falls from 0.63 to 0.51 once negatives are degree-matched",
                 fontsize=11.5, color="#334155", y=1.03, linespacing=1.5)
    fig.tight_layout()
    save(fig, "fig8_protocol_leak.svg")


# ══════════════════════════════════════════════════════════════════
# 图 9：两种协议下的完整排名对比
# ══════════════════════════════════════════════════════════════════
def fig_ab_ranking():
    cora = load("cora_protocol_ab.json")
    cite = load("citeseer_protocol_ab.json")
    cora_h = load("cora_heuristics_ab.json")
    cite_h = load("citeseer_heuristics_ab.json")

    methods = [("CN", "CN"), ("Jaccard", "Jaccard"), ("AA", "AA"),
               ("DeepWalk", "DeepWalk"), ("Node2Vec (DFS-like)", "N2V-DFS"),
               ("Ours Jaccard γ=1", "Ours γ=1"), ("Ours RA γ=1", "Ours RA"),
               ("Ours AA γ=1", "Ours AA"), ("Ours Jaccard γ=4", "Ours γ=4")]

    def val(data, heur, name, proto):
        if name in ("CN", "Jaccard", "AA"):
            return float(np.mean(heur[name][proto]))
        return float(np.mean(data[f"{name}|{proto}"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    for ax, data, heur, title in zip(axes, [cora, cite], [cora_h, cite_h],
                                     ["Cora", "Citeseer"]):
        a = [val(data, heur, k, "A") for k, _ in methods]
        b = [val(data, heur, k, "B") for k, _ in methods]
        x = np.arange(len(methods))
        ax.bar(x - 0.19, a, 0.36, color="#fda4af", edgecolor="#e11d48",
               linewidth=1.0, label="A · random negatives, raw vectors")
        ax.bar(x + 0.19, b, 0.36, color="#a5b4fc", edgecolor=INDIGO,
               linewidth=1.0, label="B · degree-matched negatives, unit-norm vectors")
        for xi in x:
            ax.text(xi - 0.19, a[xi] + 0.004, f"{a[xi]:.3f}", ha="center",
                    fontsize=7.6, color="#9f1239", rotation=90, va="bottom", alpha=.95)
            ax.text(xi + 0.19, b[xi] + 0.004, f"{b[xi]:.3f}", ha="center",
                    fontsize=7.6, color="#312e81", rotation=90, va="bottom", alpha=.95)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in methods], fontsize=9, rotation=30, ha="right")
        lo = min(min(a), min(b)) - 0.048
        ax.set_ylim(lo, max(max(a), max(b)) + 0.040)
        ax.set_title(title, fontsize=14, fontweight="bold", color="#0f172a", pad=8)
        ax.axvline(2.5, ls=":", lw=1.2, color="#94a3b8", zorder=0)
        ax.set_ylabel("AUC-ROC (10 splits)" if title == "Cora" else "")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.055))
    fig.suptitle("Method ranking under the original vs the corrected evaluation protocol",
                 fontsize=12, color="#334155", y=1.01)
    fig.tight_layout()
    save(fig, "fig9_ab_ranking.svg")


# ══════════════════════════════════════════════════════════════════
# 图 10：混合预测器（E/H/M 三种特征组合）
# ══════════════════════════════════════════════════════════════════
def fig_hybrid():
    cora = load("cora_hybrid.json")
    cite = load("citeseer_hybrid.json")
    cfgs = [("DeepWalk|A", "DeepWalk\nprotocol A"), ("DeepWalk|B", "DeepWalk\nprotocol B"),
            ("Ours Jaccard γ=1|A", "Ours γ=1\nprotocol A"), ("Ours Jaccard γ=1|B", "Ours γ=1\nprotocol B")]
    cols = [("E", "E · embedding only", "#a5b4fc", INDIGO),
            ("H", "H · structure only", "#fcd34d", AMBER),
            ("M", "M · embedding + structure", "#6ee7b7", GREEN)]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), sharey=False)
    for ax, data, title in zip(axes, [cora, cite], ["Cora", "Citeseer"]):
        x = np.arange(len(cfgs))
        w = 0.26
        for i, (tag, lbl, fc, ec) in enumerate(cols):
            vals = [float(np.mean(data[c][tag])) for c, _ in cfgs]
            errs = [float(np.std(data[c][tag], ddof=1)) for c, _ in cfgs]
            ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=2.6, color=fc,
                   edgecolor=ec, linewidth=1.0, label=lbl,
                   error_kw=dict(ecolor=ec, lw=1.0))
            if tag in ("E", "M"):
                for xi, (v, e) in zip(x, zip(vals, errs)):
                    ax.text(xi + (i - 1) * w, v + e + 0.0018, f"{v:.3f}", ha="center",
                            fontsize=6.6, va="bottom", color=ec)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in cfgs], fontsize=9)
        lo = min(np.mean(data[c][t]) for c, _ in cfgs for t, *_ in cols)
        hi = max(np.mean(data[c][t]) for c, _ in cfgs for t, *_ in cols)
        ax.set_ylim(lo - 0.014, hi + 0.022)
        ax.set_title(title, fontsize=13.5, fontweight="bold", color="#0f172a", pad=8)
        ax.set_ylabel("AUC-ROC (10 splits)" if title == "Cora" else "")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Adding structural similarity as an explicit feature gains ~0.001 AUC — "
                 "the information is already in the embedding",
                 fontsize=11, color="#334155", y=1.02)
    fig.tight_layout()
    save(fig, "fig10_hybrid.svg")

if __name__ == "__main__":
    print("🎨 生成矢量图表 ...")
    for fn in [fig_main, fig_variance, fig_single_vs_repeated, fig_variants,
               fig_ablation, fig_gantt, fig_walk_schematic,
               fig_protocol_leak, fig_ab_ranking, fig_hybrid]:
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
    print("✅ 完成")
