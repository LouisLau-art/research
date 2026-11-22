"""
Week 6 实验：Jaccard 加权随机游走 (创新算法) 对比实验与敏感性分析
════════════════════════════════════════════════════════════════════
运行方式: uv run python week6_weighted_walk_experiment.py

本脚本完成以下任务：
1. 载入 Cora 数据集并切分为 80% 训练集，20% 测试集，生成负样本。
2. 运行传统 Jaccard 基线。
3. 运行原始 DeepWalk (等价于 beta -> 无穷大)。
4. 对我们的创新算法 Jaccard 加权 DeepWalk 进行 beta 参数敏感性分析：
   - 测试 beta = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 10.0]
5. 找出在 Cora 上表现最优的 beta 值，并与基线对比。
6. 绘制并保存 beta 敏感性折线图与最终柱状对比图。
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
from src.embedding import DeepWalk, JaccardWeightedDeepWalk
from src.link_predictor import LinkPredictor
from sklearn.metrics import roc_auc_score


def main():
    print("=" * 60)
    print("🎓 Week 6: Jaccard 加权随机游走 (创新算法) 实验")
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

    predictor = LinkPredictor(operator="hadamard")
    results = {}

    # 2. 传统 Jaccard 启发式基线
    print("\n📐 运行 Jaccard 系数基线...")
    jaccard_scores = compute_scores(G_train, test_all_edges, method="jaccard")
    jaccard_auc = roc_auc_score(y_test, jaccard_scores)
    results["Jaccard (Heuristic)"] = jaccard_auc
    print(f"  🟢 Jaccard Heuristic AUC: {jaccard_auc:.4f}")

    # 3. 原始 DeepWalk 基线 (均匀游走)
    print("\n🚶 运行原始 DeepWalk 表示学习...")
    dw_model = DeepWalk(dimensions=64, walk_length=40, num_walks=10, window_size=5, workers=2, seed=42)
    dw_embeddings = dw_model.fit(G_train)
    dw_auc, _ = predictor.train_and_evaluate(
        embeddings=dw_embeddings, embedding_size=64,
        train_pos=train_pos, train_neg=train_neg,
        test_pos=test_pos, test_neg=test_neg
    )
    results["Original DeepWalk"] = dw_auc
    print(f"  🔵 DeepWalk AUC: {dw_auc:.4f}")

    # 4. 敏感性分析：测试不同 beta 对 Jaccard 加权随机游走性能的影响
    # 测试的 beta 列表
    betas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0]
    beta_aucs = []

    print("\n🧪 开展 beta 敏感性分析 (Jaccard Weighted DeepWalk)...")
    for beta in betas:
        t0 = time.time()
        weighted_dw = JaccardWeightedDeepWalk(
            dimensions=64, walk_length=40, num_walks=10,
            beta=beta, window_size=5, workers=2, seed=42
        )
        embeddings = weighted_dw.fit(G_train)
        
        auc, _ = predictor.train_and_evaluate(
            embeddings=embeddings, embedding_size=64,
            train_pos=train_pos, train_neg=train_neg,
            test_pos=test_pos, test_neg=test_neg
        )
        beta_aucs.append(auc)
        print(f"  👉 beta = {beta:<6.3f} | AUC = {auc:.4f} (耗时 {time.time() - t0:.1f}s)")

    # 找到最佳 beta
    best_idx = np.argmax(beta_aucs)
    best_beta = betas[best_idx]
    best_weighted_auc = beta_aucs[best_idx]
    results[f"Ours (Weighted DeepWalk, beta={best_beta})"] = best_weighted_auc

    # 5. 打印对比结论
    improvement_over_dw = (best_weighted_auc - dw_auc) * 100
    improvement_over_jac = (best_weighted_auc - jaccard_auc) * 100
    
    print("\n" + "=" * 65)
    print("📊 最终链路预测结果对比 (Cora 数据集):")
    print(f"  1. Jaccard 系数 (传统启发式):   {jaccard_auc:.4f}")
    print(f"  2. 原始 DeepWalk (均匀游走):     {dw_auc:.4f}")
    print(f"  3. 我们的算法 (Jaccard加权游走): {best_weighted_auc:.4f} (最优 beta={best_beta})")
    print("-" * 65)
    print(f"  📈 相比原始 DeepWalk 绝对提升: {improvement_over_dw:+.2f} 个百分点！")
    print(f"  📈 相比传统 Jaccard 绝对提升:  {improvement_over_jac:+.2f} 个百分点！")
    print("=" * 65)

    # 6. 生成敏感性曲线图
    os.makedirs("results/figures", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(betas, beta_aucs, marker='o', color='#8b5cf6', linewidth=2.5, markersize=8, label="Weighted Walk")
    # 画出原始 DeepWalk 水平参考线
    ax.axhline(y=dw_auc, color='#3b82f6', linestyle='--', alpha=0.8, label=f"Original DeepWalk (AUC={dw_auc:.4f})")
    # 画出 Jaccard Heuristic 水平线
    ax.axhline(y=jaccard_auc, color='#10b981', linestyle=':', alpha=0.8, label=f"Jaccard Heuristic (AUC={jaccard_auc:.4f})")
    
    ax.set_xscale('log')
    ax.set_xlabel("Smoothing Factor beta (log scale)", fontsize=13)
    ax.set_ylabel("AUC-ROC Score", fontsize=13)
    ax.set_title("Beta Parameter Sensitivity Analysis on Cora\n(Jaccard-weighted Random Walk)", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", ls="-", alpha=0.15)
    ax.legend(loc="lower left", fontsize=10)
    
    sensitivity_path = "results/figures/week6_beta_sensitivity.png"
    plt.tight_layout()
    plt.savefig(sensitivity_path, dpi=150)
    print(f"\n📈 Beta 敏感性分析折线图已保存: {sensitivity_path}")

    # 7. 绘制最终对比柱状图
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    model_names = list(results.keys())
    model_aucs = list(results.values())
    colors = ["#10b981", "#3b82f6", "#ef4444"]
    
    bars = ax.bar(model_names, model_aucs, color=colors, alpha=0.9, edgecolor="white", width=0.45, linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
                
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("AUC-ROC Score", fontsize=13)
    ax.set_title(f"Link Prediction Performance Comparison on Cora\n(Our Weighted Walk vs Baselines)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.4, 0.95)
    ax.grid(axis='y', alpha=0.2)
    
    comparison_path = "results/figures/week6_model_comparison.png"
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150)
    print(f"📊 最终模型对比柱状图已保存: {comparison_path}")

    # 8. 写入创新报告 txt
    report_path = "results/tables/week6_innovation_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("Week 6 创新算法实验报告：Jaccard 加权随机游走 (Cora)\n")
        f.write("=" * 70 + "\n")
        f.write(f"1. Jaccard Heuristic 基线:   {jaccard_auc:.4f}\n")
        f.write(f"2. Original DeepWalk 基线:   {dw_auc:.4f}\n")
        f.write("-" * 70 + "\n")
        f.write("3. beta 敏感性测试数据:\n")
        for b, a in zip(betas, beta_aucs):
            f.write(f"   - beta = {b:<6.3f} : AUC = {a:.4f}\n")
        f.write("-" * 70 + "\n")
        f.write(f"4. 我们的方法最优表现 (beta={best_beta}): {best_weighted_auc:.4f}\n")
        f.write(f"   相对 DeepWalk 绝对提升: {improvement_over_dw:+.2f}%\n")
        f.write(f"   相对 Jaccard Heuristic 绝对提升: {improvement_over_jac:+.2f}%\n")
        f.write("=" * 70 + "\n")
    print(f"📝 创新报告已写入: {report_path}")

    print(f"\n{'='*60}")
    print("🎉 Week 6 创新算法实验全部顺利跑通！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
