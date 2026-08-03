"""
Week 1 探索脚本：认识图、认识数据集
═══════════════════════════════════
运行方式: uv run python week1_explore.py

这个脚本会带你完成：
1. 下载 Cora 数据集
2. 把它加载成一个"图"
3. 打印基本统计信息
4. 画出一小块图的样子
5. 理解"邻居"和"度"的概念
"""

import sys
import os

# 让 Python 能找到 src 目录
sys.path.insert(0, os.path.dirname(__file__))

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式，直接保存文件
import matplotlib.pyplot as plt
from collections import Counter

from src.graph_utils import download_dataset, load_cora, print_graph_stats, get_degree_distribution


def main():
    print("=" * 60)
    print("🎓 Week 1: 认识图 (Graph) 和链路预测 (Link Prediction)")
    print("=" * 60)

    # ──────────────────────────────────────────────
    # Step 1: 下载并加载 Cora 数据集
    # ──────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("📥 Step 1: 下载并加载 Cora 引文网络")
    print("─" * 50)
    print("""
    Cora 是什么？
    ─────────────
    想象一下：有 2708 篇计算机科学论文，每篇论文会引用其他论文。
    我们把每篇论文看成一个"点"（节点），
    如果论文 A 引用了论文 B，我们就在 A 和 B 之间画一条线（边）。
    
    这就构成了一个"图"（Graph）！
    
    你的毕业论文做的是"链路预测"：
    → 如果我把一些连线藏起来，你能猜到哪些点之间本来是有连线的吗？
    → 这就像预测"两个人将来会不会成为朋友"
    """)

    download_dataset("cora")
    G, features, labels = load_cora()

    # ──────────────────────────────────────────────
    # Step 2: 看看这个图长什么样（统计信息）
    # ──────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("📊 Step 2: 图的基本统计信息")
    print("─" * 50)
    print("""
    理解这些统计量：
    ────────────────
    • 节点数 = 网络中有多少篇论文
    • 边数   = 有多少对引用关系
    • 度     = 一个节点有几条边（= 一篇论文引用/被引了几次）
    • 密度   = 实际边数 ÷ 理论最大边数
              如果所有人都互相认识，密度 = 1
              现实社交网络的密度通常非常低（<0.01）
    """)

    stats = print_graph_stats(G, "Cora")

    # ──────────────────────────────────────────────
    # Step 3: 看看节点的类别分布
    # ──────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("🏷️  Step 3: 论文类别分布")
    print("─" * 50)

    label_counts = Counter(labels.values())
    print(f"\n  Cora 的论文被分为 {len(label_counts)} 个类别：")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 20)
        print(f"    {label:30s}  {count:4d} 篇  {bar}")

    # ──────────────────────────────────────────────
    # Step 4: 理解"邻居"的概念
    # ──────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("👥 Step 4: 理解'邻居'（Neighbors）")
    print("─" * 50)
    print("""
    邻居 = 和你直接相连的节点
    
    在社交网络里：邻居 = 你的好友
    在引文网络里：邻居 = 引用你的论文 + 你引用的论文
    
    让我们看几个具体节点的邻居情况：
    """)

    # 找几个有代表性的节点
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: -x[1])

    # 度最高的节点（"大V"）
    top_node, top_deg = sorted_nodes[0]
    print(f"  🌟 度最高的节点: {top_node}")
    print(f"     度 = {top_deg}（它和 {top_deg} 个其他节点相连）")
    print(f"     类别: {labels[top_node]}")
    neighbors = list(G.neighbors(top_node))
    print(f"     前 5 个邻居: {neighbors[:5]}")

    # 度一般的节点
    mid_idx = len(sorted_nodes) // 2
    mid_node, mid_deg = sorted_nodes[mid_idx]
    print(f"\n  📌 度中等的节点: {mid_node}")
    print(f"     度 = {mid_deg}")
    print(f"     类别: {labels[mid_node]}")

    # 度最低的节点
    low_node, low_deg = sorted_nodes[-1]
    print(f"\n  📎 度最低的节点: {low_node}")
    print(f"     度 = {low_deg}")
    print(f"     类别: {labels[low_node]}")

    # ──────────────────────────────────────────────
    # Step 5: 理解"共同邻居"
    # ──────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("🤝 Step 5: 理解'共同邻居'（Common Neighbors）")
    print("─" * 50)
    print("""
    共同邻居 = 两个人共同认识的朋友
    
    直觉：如果你和我有 10 个共同好友，
    那我们成为好友的概率，比只有 1 个共同好友的两个人高得多。
    
    这就是链路预测最朴素的想法！
    """)

    # 找一条实际存在的边，看看它的共同邻居
    sample_edge = list(G.edges())[0]
    u, v = sample_edge
    common = sorted(nx.common_neighbors(G, u, v))
    print(f"  节点 {u} 和节点 {v} 之间有一条边")
    print(f"  它们的共同邻居有 {len(common)} 个: {common[:10]}")
    print(f"  Jaccard 系数: {list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]:.4f}")

    # 找一对没有边的节点对
    for u2 in list(G.nodes())[:50]:
        for v2 in list(G.nodes())[50:100]:
            if not G.has_edge(u2, v2) and u2 != v2:
                common2 = sorted(nx.common_neighbors(G, u2, v2))
                if len(common2) > 0:
                    print(f"\n  节点 {u2} 和节点 {v2} 之间没有边")
                    print(f"  但它们有 {len(common2)} 个共同邻居: {common2[:10]}")
                    print(f"  Jaccard 系数: {list(nx.jaccard_coefficient(G, [(u2, v2)]))[0][2]:.4f}")
                    print(f"  → 也许它们之间「应该」有一条边？这就是链路预测在做的事！")
                    break
        else:
            continue
        break

    # ──────────────────────────────────────────────
    # Step 6: 可视化
    # ──────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("🎨 Step 6: 画出 Cora 图的一小块")
    print("─" * 50)

    # 6a: 度分布图
    print("  📈 生成度分布图...")
    degrees_list, counts = get_degree_distribution(G)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 线性坐标
    axes[0].bar(degrees_list, counts, color="#4f46e5", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Degree (度)", fontsize=12)
    axes[0].set_ylabel("Count (数量)", fontsize=12)
    axes[0].set_title("Cora Degree Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlim(0, 50)

    # 对数坐标（更能看出幂律分布）
    axes[1].scatter(degrees_list, counts, color="#e11d48", alpha=0.7, s=30)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Degree (log scale)", fontsize=12)
    axes[1].set_ylabel("Count (log scale)", fontsize=12)
    axes[1].set_title("Cora Degree Distribution (Log-Log)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/figures/cora_degree_distribution.png", dpi=150, bbox_inches="tight")
    print("     → 已保存: results/figures/cora_degree_distribution.png")

    # 6b: 子图可视化（取度最高节点的 2-hop 邻域）
    print("  🔍 生成子图可视化...")

    # 取度最高节点的 1-hop 邻居（限制在 50 个节点以内）
    center_node = sorted_nodes[0][0]
    ego_graph = nx.ego_graph(G, center_node, radius=1)

    # 如果太大就只保留前 50 个节点
    if len(ego_graph) > 60:
        nodes_to_keep = [center_node] + list(G.neighbors(center_node))[:59]
        ego_graph = ego_graph.subgraph(nodes_to_keep).copy()

    # 给不同类别染不同颜色
    color_map = {}
    unique_labels = sorted(set(labels.values()))
    colors_palette = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    for i, lab in enumerate(unique_labels):
        color_map[lab] = colors_palette[i]

    node_colors = []
    node_sizes = []
    for node in ego_graph.nodes():
        node_colors.append(color_map.get(labels.get(node, "Unknown"), [0.5, 0.5, 0.5, 1.0]))
        if node == center_node:
            node_sizes.append(500)  # 中心节点画大一些
        else:
            node_sizes.append(120)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    pos = nx.spring_layout(ego_graph, seed=42, k=0.8)

    nx.draw_networkx_edges(ego_graph, pos, alpha=0.2, width=0.5, ax=ax)
    nx.draw_networkx_nodes(ego_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)

    # 只给中心节点标注 ID
    nx.draw_networkx_labels(ego_graph, pos, labels={center_node: str(center_node)},
                            font_size=10, font_weight="bold", ax=ax)

    ax.set_title(f"Cora: Node {center_node}'s Neighborhood (Ego Graph)\n"
                 f"Center node degree = {G.degree(center_node)}, "
                 f"subgraph size = {len(ego_graph)} nodes",
                 fontsize=13, fontweight="bold")

    # 图例
    legend_elements = []
    for lab, col in sorted(color_map.items()):
        from matplotlib.lines import Line2D
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=col, markersize=8, label=lab))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("results/figures/cora_ego_graph.png", dpi=150, bbox_inches="tight")
    print("     → 已保存: results/figures/cora_ego_graph.png")

    # ──────────────────────────────────────────────
    # Step 7: 演示"链路预测"的基本流程
    # ──────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("🔮 Step 7: 链路预测的完整流程（预告）")
    print("─" * 50)
    print("""
    接下来几周，你要做的事情可以用一句话概括：
    
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   把图的一些边藏起来 → 用剩下的边训练模型        │
    │   → 让模型预测被藏起来的边 → 看猜得准不准        │
    │                                                  │
    └──────────────────────────────────────────────────┘
    
    评价指标是 AUC-ROC（0.5 = 瞎猜，1.0 = 完美预测）
    
    Week 2: 用 Jaccard/Adamic-Adar 等经典指标做预测
    Week 3: 理解 Word2Vec（DeepWalk 的核心引擎）
    Week 4: 用 DeepWalk 做预测（随机游走 + Word2Vec）
    Week 5: 用 Node2Vec 和 LINE 做预测
    Week 6: 实现你的创新——Jaccard 加权随机游走
    Week 7-8: 三个数据集跑全部实验 + 写论文
    """)

    from src.graph_utils import split_edges
    G_train, test_pos, test_neg = split_edges(G)

    print(f"\n  📝 小测验时间（检验你是否理解了）：")
    print(f"     Q1: Cora 有 {stats['nodes']} 个节点，{stats['edges']} 条边。")
    print(f"         平均每个节点和 {stats['avg_degree']:.1f} 个节点相连。")
    print(f"     Q2: 如果我藏起 20% 的边，就是 {len(test_pos)} 条边。")
    print(f"         模型需要从 {len(test_pos)} 条真边 + {len(test_neg)} 条假边")
    print(f"         中分辨出哪些是真的。")
    print(f"     Q3: 图的密度只有 {stats['density']:.6f}，说明这个网络非常稀疏")
    print(f"         ——绝大多数论文之间并没有引用关系。")

    print(f"\n{'='*60}")
    print(f"✅ Week 1 探索完成！")
    print(f"   查看生成的可视化图片：")
    print(f"   📊 results/figures/cora_degree_distribution.png")
    print(f"   🔍 results/figures/cora_ego_graph.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
