"""
Week 5 实验：Node2Vec 链路预测与对比实验
═════════════════════════════════════════════
运行方式: uv run python week5_node2vec_experiment.py

本脚本完成以下任务：
1. 载入 Cora 数据集并切分为 80% 训练集，20% 测试集，生成负样本。
2. 运行 Jaccard 系数基线。
3. 运行 DeepWalk 表示学习基线。
4. 运行有偏游走的 Node2Vec 算法：
   - 模式 A (BFS-like): p=0.5, q=2.0
   - 模式 B (DFS-like): p=2.0, q=0.5
5. 用二分类器和 Hadamard 积评测各模型的 AUC。
6. 绘制多模型柱状对比图，保存结果报告。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from src.graph_utils import download_dataset, load_cora, print_graph_stats, split_edges
from src.similarity import compute_scores
from src.embedding import DeepWalk, Node2Vec
from src.link_predictor import LinkPredictor
from sklearn.metrics import roc_auc_score


def main():
    print("=" * 60)
    print("🎓 Week 5: Node2Vec 链路预测对比实验")
    print("=" * 60)

    # 1. 载入数据与边切分
    download_dataset("cora")
    G, _, _ = load_cora()
    G_train, test_pos, test_neg = split_edges(G, test_ratio=0.2, seed=42)

    # 训练负样本（分类器训练用）
    train_pos = list(G_train.edges())
    non_edges = list(nx.non_edges(G_train))
    rng = np.random.RandomState(42)
    rng.shuffle(non_edges)
    train_neg = non_edges[:len(train_pos)]

    test_all_edges = test_pos + test_neg
    y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))

    results = {}

    # 2. 传统 Jaccard 基线
    print("\n📐 运行 Jaccard 系数基线...")
    t0 = time.time()
    jaccard_scores = compute_scores(G_train, test_all_edges, method="jaccard")
    jaccard_auc = roc_auc_score(y_test, jaccard_scores)
    results["Jaccard"] = jaccard_auc
    print(f"  🟢 Jaccard AUC: {jaccard_auc:.4f} (耗时 {time.time() - t0:.3f}s)")

    # 3. 运行 DeepWalk 基线
    print("\n🚶 运行 DeepWalk 表示学习...")
    dw_model = DeepWalk(dimensions=64, walk_length=40, num_walks=10, window_size=5, workers=2, seed=42)
    dw_embeddings = dw_model.fit(G_train)
    
    predictor = LinkPredictor(operator="hadamard")
    dw_auc, _ = predictor.train_and_evaluate(
        embeddings=dw_embeddings, embedding_size=64,
        train_pos=train_pos, train_neg=train_neg,
        test_pos=test_pos, test_neg=test_neg
    )
    results["DeepWalk"] = dw_auc
    print(f"  🔵 DeepWalk AUC: {dw_auc:.4f}")

    # 4. 运行 Node2Vec (模式 A: BFS-like, p=0.5, q=2.0)
    print("\n🚀 运行 Node2Vec (BFS-like: p=0.5, q=2.0)...")
    n2v_bfs = Node2Vec(dimensions=64, walk_length=40, num_walks=10, p=0.5, q=2.0, window_size=5, workers=2, seed=42)
    bfs_embeddings = n2v_bfs.fit(G_train)
    
    bfs_auc, _ = predictor.train_and_evaluate(
        embeddings=bfs_embeddings, embedding_size=64,
        train_pos=train_pos, train_neg=train_neg,
        test_pos=test_pos, test_neg=test_neg
    )
    results["Node2Vec (BFS-like)"] = bfs_auc
    print(f"  🟣 Node2Vec (p=0.5, q=2.0) AUC: {bfs_auc:.4f}")

    # 5. 运行 Node2Vec (模式 B: DFS-like, p=2.0, q=0.5)
    print("\n🚀 运行 Node2Vec (DFS-like: p=2.0, q=0.5)...")
    n2v_dfs = Node2Vec(dimensions=64, walk_length=40, num_walks=10, p=2.0, q=0.5, window_size=5, workers=2, seed=42)
    dfs_embeddings = n2v_dfs.fit(G_train)
    
    dfs_auc, _ = predictor.train_and_evaluate(
        embeddings=dfs_embeddings, embedding_size=64,
        train_pos=train_pos, train_neg=train_neg,
        test_pos=test_pos, test_neg=test_neg
    )
    results["Node2Vec (DFS-like)"] = dfs_auc
    print(f"  🔴 Node2Vec (p=2.0, q=0.5) AUC: {dfs_auc:.4f}")

    # 6. 实验结论输出
    print("\n" + "=" * 60)
    print("📊 最终链路预测结果对比 (Cora 数据集):")
    for name, auc in results.items():
        print(f"  - {name:25s}: AUC = {auc:.4f}")
    print("=" * 60)

    # 7. 绘制并保存柱状图
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    names = list(results.keys())
    aucs = list(results.values())
    colors = ["#10b981", "#3b82f6", "#8b5cf6", "#ef4444"]
    
    bars = ax.bar(names, aucs, color=colors, alpha=0.85, edgecolor="white", width=0.5, linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
                
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.set_ylabel("AUC-ROC Score", fontsize=13)
    ax.set_title("Link Prediction Performance Comparison on Cora\n(Classical vs DeepWalk vs Node2Vec)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0.4, 0.95)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(loc="lower right")
    
    save_path = "results/figures/week5_comparison.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 对比结果柱状图已保存至: {save_path}")

    # 8. 写入 txt 报告
    report_path = "results/tables/week5_auc_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("Week 5 链路预测对比报告：Jaccard vs DeepWalk vs Node2Vec (Cora)\n")
        f.write("=" * 70 + "\n")
        f.write(f"1. Jaccard Coefficient:              {results['Jaccard']:.4f}\n")
        f.write(f"2. Original DeepWalk:                {results['DeepWalk']:.4f}\n")
        f.write(f"3. Node2Vec (BFS-like, p=0.5, q=2.0): {results['Node2Vec (BFS-like)']:.4f}\n")
        f.write(f"4. Node2Vec (DFS-like, p=2.0, q=0.5): {results['Node2Vec (DFS-like)']:.4f}\n")
        f.write("=" * 70 + "\n")
    print(f"📝 结果报告已写入至: {report_path}")

    print(f"\n{'='*60}")
    print("🎉 Week 5 Node2Vec 实验全部顺利跑通！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
