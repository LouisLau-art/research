"""
Week 4 实验：DeepWalk vs 经典指标 链路预测对比实验
═════════════════════════════════════════════════════
运行方式: uv run python week4_deepwalk_experiment.py

本脚本完成以下任务：
1. 载入 Cora 数据集并切分为 80% 训练集，20% 测试集。
2. 提取测试集的训练负样本（用于二分类器训练）和测试负样本。
3. 计算经典 Jaccard 指标作为对比基线。
4. 训练 DeepWalk 算法，获得 64 维节点嵌入。
5. 通过 Hadamard 算子融合边特征，并用 Logistic Regression 做链路预测。
6. 绘制并保存 AUC 柱状对比图，打印提升百分点。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import networkx as nx

from src.graph_utils import download_dataset, load_cora, print_graph_stats, split_edges
from src.similarity import compute_scores
from src.embedding import DeepWalk
from src.link_predictor import LinkPredictor
from sklearn.metrics import roc_auc_score


def main():
    print("=" * 60)
    print("🎓 Week 4: DeepWalk 链路预测 Pipeline 与对比实验")
    print("=" * 60)

    # 1. 数据集加载与处理
    download_dataset("cora")
    G, _, _ = load_cora()
    print_graph_stats(G, "Cora")

    # 2. 分割边（80% 训练，20% 测试）
    # 注意：为了让模型能学到怎么“区分负样本”，我们不仅需要测试集的负边，
    # 在基于表示学习的模型中，训练分类器本身也需要训练集的正/负样本。
    print("\n✂️  正在切分图的边...")
    G_train, test_pos, test_neg = split_edges(G, test_ratio=0.2, seed=42)

    # 生成训练集的负样本（用不存在的边，数量等同于训练集的正边）
    # 这是训练分类器必需的
    print("  生成训练负样本（分类器训练用）...")
    train_pos = list(G_train.edges())
    non_edges = list(nx.non_edges(G_train))
    rng = np.random.RandomState(42)
    rng.shuffle(non_edges)
    train_neg = non_edges[:len(train_pos)]
    print(f"  训练正样本数: {len(train_pos)}, 训练负样本数: {len(train_neg)}")

    # 3. 运行 Jaccard 基线
    print("\n📐 计算 Jaccard 系数基线...")
    test_all_edges = test_pos + test_neg
    y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))
    
    t0 = time.time()
    jaccard_scores = compute_scores(G_train, test_all_edges, method="jaccard")
    jaccard_auc = roc_auc_score(y_test, jaccard_scores)
    jaccard_time = time.time() - t0
    print(f"  🟢 Jaccard AUC: {jaccard_auc:.4f} (耗时 {jaccard_time:.3f}s)")

    # 4. 运行 DeepWalk 表示学习
    # 超参数：dimensions=64, walk_length=40, num_walks=10, window_size=5
    print("\n🚶 正在训练 DeepWalk 节点表示...")
    dw_model = DeepWalk(
        dimensions=64,
        walk_length=40,
        num_walks=10,
        window_size=5,
        workers=2,
        seed=42
    )
    t0 = time.time()
    embeddings = dw_model.fit(G_train)
    dw_train_time = time.time() - t0
    print(f"  🚶 DeepWalk 训练耗时: {dw_train_time:.2f}s")

    # 5. 使用 LinkPredictor 训练分类器并预测
    print("\n🔮 正在提取边特征并训练链路预测器...")
    predictor = LinkPredictor(operator="hadamard")
    t0 = time.time()
    dw_auc, dw_ap = predictor.train_and_evaluate(
        embeddings=embeddings,
        embedding_size=64,
        train_pos=train_pos,
        train_neg=train_neg,
        test_pos=test_pos,
        test_neg=test_neg
    )
    dw_eval_time = time.time() - t0
    print(f"  🔵 DeepWalk + LR (Hadamard) AUC: {dw_auc:.4f} (预测耗时 {dw_eval_time:.3f}s)")
    print(f"  🔵 DeepWalk + LR (Hadamard) AP : {dw_ap:.4f}")

    # 6. 对比与柱状图绘制
    improvement = (dw_auc - jaccard_auc) * 100
    print("\n" + "=" * 55)
    print("📊 链路预测结果对比 (Cora 数据集):")
    print(f"  1. Jaccard 系数 (传统指标):  {jaccard_auc:.4f}")
    print(f"  2. DeepWalk 表示学习 (Hadamard): {dw_auc:.4f}")
    print(f"  📈 提升效果: {improvement:+.2f} 个百分点！")
    print("=" * 55)

    # 绘制柱状图
    os.makedirs("results/figures", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    methods = ["Jaccard (Classical)", "DeepWalk (Representation)"]
    aucs = [jaccard_auc, dw_auc]
    colors = ["#10b981", "#3b82f6"] # 绿色与蓝色
    
    bars = ax.bar(methods, aucs, color=colors, alpha=0.9, edgecolor="white", width=0.5, linewidth=2)
    
    # 标注数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
                
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.set_ylabel("AUC-ROC Score", fontsize=13)
    ax.set_title(f"Link Prediction Performance Comparison on Cora\n(DeepWalk vs Jaccard, Improvement = {improvement:+.2f}%)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(loc="lower right")
    
    save_path = "results/figures/week4_comparison.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 对比结果柱状图已保存至: {save_path}")

    # 7. 写入最终 txt 报告
    report_path = "results/tables/week4_deepwalk_vs_jaccard.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("Week 4 链路预测对比报告：DeepWalk vs Jaccard (Cora)\n")
        f.write("=" * 60 + "\n")
        f.write(f"训练图节点数: {G_train.number_of_nodes()}, 边数: {G_train.number_of_edges()}\n")
        f.write(f"测试集正样本: {len(test_pos)} 条, 负样本: {len(test_neg)} 条\n")
        f.write("-" * 60 + "\n")
        f.write(f"Jaccard (传统拓扑指标) AUC-ROC:  {jaccard_auc:.4f}\n")
        f.write(f"DeepWalk (随机游走表示学习) AUC-ROC: {dw_auc:.4f}\n")
        f.write(f"性能提升 (Absolute Improvement):  {improvement:+.2f}%\n")
        f.write("=" * 60 + "\n")
    print(f"📝 结果报告已写入至: {report_path}")

    print(f"\n{'='*60}")
    print("🎉 Week 4 实验全部顺利跑通！")
    print("   你已经建立了一个严谨、自动化的 Graph Embedding 链路预测框架。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
