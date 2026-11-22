"""
Week 2 实验：经典相似度指标链路预测 + 随机游走演示
════════════════════════════════════════════════════
运行方式: uv run python week2_similarity_experiment.py

本脚本完成三件事：
1. 用四种经典指标做链路预测，拿到 AUC 分数
2. 演示随机游走的过程
3. 生成对比图表

这是你的第一个正式实验！跑完后你会知道：
"在什么都不用 AI 的情况下，纯靠数学公式能预测到什么程度？"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import time

from src.graph_utils import download_dataset, load_cora, print_graph_stats, split_edges
from src.similarity import compute_scores, demo_compare_metrics
from src.random_walk import demo_random_walk, generate_walks


def run_link_prediction_experiment(G_train, test_pos_edges, test_neg_edges):
    """
    用四种经典指标做链路预测实验。

    实验流程：
    ─────────
    1. 对每条测试边（正样本 + 负样本），计算相似度分数
    2. 分数越高 → 模型越认为这条边存在
    3. 用 AUC-ROC 评估：分数高的是不是真的是正样本？

    AUC 的含义：
    ───────────
    - AUC = 0.5 → 和瞎猜一样
    - AUC = 0.7 → 还行，有一定预测能力
    - AUC = 0.8 → 不错！
    - AUC = 0.9 → 很好！
    - AUC = 1.0 → 完美预测（不可能达到）
    """
    # 准备标签：正样本 = 1，负样本 = 0
    all_edges = list(test_pos_edges) + list(test_neg_edges)
    y_true = np.array([1] * len(test_pos_edges) + [0] * len(test_neg_edges))

    methods = {
        "Common Neighbors": "cn",
        "Jaccard":          "jaccard",
        "Adamic-Adar":      "aa",
        "Resource Alloc.":  "ra",
    }

    results = {}

    print(f"\n{'═'*60}")
    print(f"🔬 链路预测实验：经典相似度指标")
    print(f"{'═'*60}")
    print(f"  测试正样本: {len(test_pos_edges)} 条真实存在的边")
    print(f"  测试负样本: {len(test_neg_edges)} 条不存在的边")
    print(f"  任务: 用相似度分数区分哪些是真边、哪些是假边\n")

    for name, method_key in methods.items():
        t0 = time.time()
        scores = compute_scores(G_train, all_edges, method=method_key)
        elapsed = time.time() - t0

        auc = roc_auc_score(y_true, scores)
        results[name] = {
            "auc": auc,
            "scores": scores,
            "time": elapsed,
        }

        # 输出结果
        bar_len = int(auc * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        status = "🟢" if auc > 0.7 else "🟡" if auc > 0.6 else "🔴"
        print(f"  {status} {name:20s}  AUC = {auc:.4f}  [{bar}]  ({elapsed:.2f}s)")

    # 找出最优方法
    best_method = max(results.items(), key=lambda x: x[1]["auc"])
    print(f"\n  🏆 最优方法: {best_method[0]} (AUC = {best_method[1]['auc']:.4f})")

    return results, y_true


def plot_roc_curves(results, y_true, save_path="results/figures/week2_roc_curves.png"):
    """画 ROC 曲线对比图"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    colors = ["#4f46e5", "#e11d48", "#059669", "#d97706"]

    for (name, data), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, data["scores"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC = {data['auc']:.4f})")

    # 对角线（随机基线）
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("Link Prediction: ROC Curves of Classical Methods\n(Cora Dataset)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  📈 ROC 曲线已保存: {save_path}")


def plot_auc_bar_chart(results, save_path="results/figures/week2_auc_comparison.png"):
    """画 AUC 柱状图"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    names = list(results.keys())
    aucs = [results[n]["auc"] for n in names]
    colors = ["#4f46e5", "#e11d48", "#059669", "#d97706"]

    bars = ax.bar(names, aucs, color=colors, alpha=0.85, edgecolor="white", linewidth=2)

    # 在柱子上标注数值
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{auc:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # 画一条 AUC = 0.5 的基线
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline (0.5)")

    ax.set_ylabel("AUC-ROC", fontsize=13)
    ax.set_title("Link Prediction: AUC Comparison of Classical Methods\n(Cora Dataset)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0.4, max(aucs) + 0.08)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 AUC 柱状图已保存: {save_path}")


def demonstrate_random_walk_connection(G):
    """
    演示随机游走和链路预测之间的关系。
    展示为什么经常"共游"的节点更可能有边。
    """
    print(f"\n{'═'*60}")
    print(f"🚶 随机游走 × 链路预测：它们是怎么联系起来的？")
    print(f"{'═'*60}")
    print("""
    我们刚才用 Jaccard 等指标做链路预测，AUC 大概在 0.6-0.8。
    
    能不能做得更好？
    
    关键问题是：Jaccard 只看了"1-hop 邻居"（直接好友），
    完全忽略了"2-hop、3-hop 甚至更远"的结构信息。
    
    ┌───────────────────────────────────────────┐
    │  Jaccard 能看到的：                       │
    │    A ── ? ── B                            │
    │    只看 A 和 B 的直接邻居有没有重叠       │
    │                                           │
    │  随机游走能看到的：                       │
    │    A ── X ── Y ── Z ── B                  │
    │    通过"走过去"，能发现更远距离的关系     │
    └───────────────────────────────────────────┘
    
    下周（Week 3-4），我们会把随机游走 + Word2Vec 结合，
    让模型能看到更远距离的结构信息 → AUC 更高！
    """)

    # 演示一次随机游走
    demo_random_walk(G, walk_length=15)

    # 生成少量游走，展示"共现频率"
    print(f"\n{'─'*60}")
    print(f"📊 共现频率分析（100 条游走）")
    print(f"{'─'*60}")

    walks = generate_walks(G, num_walks=2, walk_length=20, seed=42)

    # 统计共现频率（在同一条游走中出现的节点对）
    cooccur = {}
    for walk in walks[:200]:
        nodes_in_walk = set(walk)
        for n1 in nodes_in_walk:
            for n2 in nodes_in_walk:
                if n1 < n2:
                    pair = (n1, n2)
                    cooccur[pair] = cooccur.get(pair, 0) + 1

    # 看看共现频率高的节点对是不是真的有边
    top_pairs = sorted(cooccur.items(), key=lambda x: -x[1])[:10]
    print(f"\n  共现频率最高的 10 对节点：")
    print(f"  {'节点对':>20s}  {'共现次数':>8s}  {'有边?':>6s}")
    print(f"  {'─'*40}")
    for (u, v), count in top_pairs:
        has_edge = "✅ 是" if G.has_edge(u, v) else "❌ 否"
        print(f"  ({u:>7d}, {v:>7d})  {count:>8d}  {has_edge}")

    has_edge_count = sum(1 for (u, v), _ in top_pairs if G.has_edge(u, v))
    print(f"\n  💡 在共现频率最高的 10 对中，{has_edge_count}/10 对确实有边！")
    print(f"     → 共现频率越高 ≈ 越可能有边 → 这就是 DeepWalk 的核心直觉")


def save_results_table(results, save_path="results/tables/week2_classical_auc.txt"):
    """保存实验结果到文本文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("Week 2 实验结果：经典相似度指标链路预测 (Cora)\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'方法':20s}  {'AUC-ROC':>10s}  {'耗时':>8s}\n")
        f.write("-" * 50 + "\n")
        for name, data in results.items():
            f.write(f"{name:20s}  {data['auc']:>10.4f}  {data['time']:>7.2f}s\n")
        f.write("=" * 50 + "\n")
    print(f"  📝 结果表格已保存: {save_path}")


def main():
    print("=" * 60)
    print("🎓 Week 2: 经典相似度指标 + 随机游走")
    print("=" * 60)

    # Step 1: 加载数据
    print("\n📥 加载 Cora 数据集...")
    download_dataset("cora")
    G, features, labels = load_cora()
    stats = print_graph_stats(G, "Cora")

    # Step 2: 分割边
    print("\n✂️  分割训练/测试集...")
    G_train, test_pos, test_neg = split_edges(G, test_ratio=0.2, seed=42)

    # Step 3: 在几对节点上演示四种指标的区别
    print("\n📐 四种指标对比演示...")
    # 取一些有共同邻居的边和没有共同邻居的边
    sample_edges = list(G.edges())[:3]
    # 找几对非邻居节点但有共同邻居
    import networkx as nx
    for u in list(G.nodes())[:100]:
        for v in list(G.nodes())[100:200]:
            if not G.has_edge(u, v) and len(list(nx.common_neighbors(G, u, v))) > 2:
                sample_edges.append((u, v))
                break
        if len(sample_edges) >= 5:
            break

    demo_compare_metrics(G_train, sample_edges)

    # Step 4: 跑链路预测实验
    results, y_true = run_link_prediction_experiment(G_train, test_pos, test_neg)

    # Step 5: 画图
    print(f"\n{'─'*60}")
    print(f"📊 生成可视化图表...")
    print(f"{'─'*60}")
    plot_roc_curves(results, y_true)
    plot_auc_bar_chart(results)
    save_results_table(results)

    # Step 6: 随机游走演示
    demonstrate_random_walk_connection(G)

    # 总结
    print(f"\n{'═'*60}")
    print(f"📋 Week 2 总结")
    print(f"{'═'*60}")
    print(f"""
    ✅ 你已经完成了第一个正式的链路预测实验！

    关键发现：
    1. 纯靠数学公式（Jaccard/AA/RA），AUC 大约在 0.6-0.8 之间
    2. 这些方法只看"1-hop 邻居"，有局限性
    3. 随机游走能捕获更远距离的结构信息

    下一步（Week 3-4）：
    → 学习 Word2Vec，理解 DeepWalk 的"AI 引擎"
    → 用 DeepWalk 做链路预测，看看 AUC 能提高多少
    → 那就是你论文中"表示学习方法"的基线！

    生成的文件：
    📈 results/figures/week2_roc_curves.png
    📊 results/figures/week2_auc_comparison.png
    📝 results/tables/week2_classical_auc.txt
    """)


if __name__ == "__main__":
    main()
