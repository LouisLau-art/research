"""
Week 7-8 终极汇总实验：多数据集大评测与消融实验 (Table & Figures)
══════════════════════════════════════════════════════════════════════
运行方式: uv run python week7_8_final_experiment.py

本脚本自动化完成毕业论文全部核心实验图表生成：
1. 三个数据集 (Cora, Citeseer, Facebook) × 七种算法的大表测试。
2. 我们的创新算法在 Cora 上的消融实验 (Embedding维度、随机游走长度)。
3. 生成论文可以直接复制的 LaTeX/Markdown 结果表格与学术风格折线图。
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

from src.graph_utils import download_dataset, load_cora, load_citeseer, load_facebook, print_graph_stats, split_edges
from src.similarity import compute_scores
from src.embedding import DeepWalk, Node2Vec, JaccardWeightedDeepWalk
from src.link_predictor import LinkPredictor
from sklearn.metrics import roc_auc_score


def run_cross_dataset_experiments():
    """
    【实验一】三个真实数据集上的 7 种算法横向大评测 (Table 1)
    """
    print("\n" + "=" * 65)
    print("🚀 正在启动：【实验一】三数据集大对比测试 (Cora, Citeseer, Facebook)")
    print("=" * 65)

    datasets = ["cora", "citeseer", "facebook"]
    methods = [
        "Common Neighbors",
        "Jaccard Coefficient",
        "Adamic-Adar",
        "DeepWalk",
        "Node2Vec (BFS-like)",
        "Node2Vec (DFS-like)",
        "Ours (Weighted Walk)"
    ]

    # 保存最终大表数据 {dataset_name: {method_name: auc_score}}
    table_data = {d: {} for d in datasets}

    for dname in datasets:
        print(f"\n📥 正在载入并切分 {dname.upper()} 数据集...")
        download_dataset(dname)
        
        if dname == "cora":
            G, _, _ = load_cora()
        elif dname == "citeseer":
            G, _, _ = load_citeseer()
        else:
            G, _, _ = load_facebook()
            
        print_graph_stats(G, dname.upper())

        # 划分训练集/测试集
        # 对于规模较大的 facebook (8.8万条边)，为了训练速度我们将 test_ratio 设为 20%
        # 同时为了避免正负样本过多导致 LR 训练过慢，限制训练正样本最多为 10000 条 (采样子图或采样边)
        G_train, test_pos, test_neg = split_edges(G, test_ratio=0.2, seed=42)
        
        train_pos = list(G_train.edges())
        if len(train_pos) > 10000:
            print(f"     💡 训练边数 ({len(train_pos)}) 较多，随机采样 10,000 条用于分类器训练...")
            rng_idx = np.random.RandomState(42)
            indices = rng_idx.choice(len(train_pos), 10000, replace=False)
            train_pos = [train_pos[i] for i in indices]
            
        # 生成训练负样本
        non_edges = list(nx.non_edges(G_train))
        rng_neg = np.random.RandomState(42)
        rng_neg.shuffle(non_edges)
        train_neg = non_edges[:len(train_pos)]

        test_all_edges = test_pos + test_neg
        y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))

        # ----------------- 1. Heuristic Methods -----------------
        print(f"  📐 计算启发式指标...")
        cn_scores = compute_scores(G_train, test_all_edges, method="cn")
        table_data[dname]["Common Neighbors"] = roc_auc_score(y_test, cn_scores)

        jac_scores = compute_scores(G_train, test_all_edges, method="jaccard")
        table_data[dname]["Jaccard Coefficient"] = roc_auc_score(y_test, jac_scores)

        aa_scores = compute_scores(G_train, test_all_edges, method="aa")
        table_data[dname]["Adamic-Adar"] = roc_auc_score(y_test, aa_scores)

        # ----------------- 2. Representation Learning -----------------
        # 对于较大型的 Facebook 图，我们略微缩减 walk 数量以极大节省 CPU 训练耗时
        num_walks = 5 if dname == "facebook" else 10
        walk_length = 30 if dname == "facebook" else 40

        predictor = LinkPredictor(operator="hadamard")

        # A. DeepWalk
        print(f"  🚶 训练 DeepWalk 基线...")
        dw = DeepWalk(dimensions=64, walk_length=walk_length, num_walks=num_walks, window_size=5, workers=2, seed=42)
        dw_embs = dw.fit(G_train)
        dw_auc, _ = predictor.train_and_evaluate(dw_embs, 64, train_pos, train_neg, test_pos, test_neg)
        table_data[dname]["DeepWalk"] = dw_auc

        # B. Node2Vec (BFS-like)
        print(f"  🟣 训练 Node2Vec (BFS-like: p=0.5, q=2.0)...")
        n2v_bfs = Node2Vec(dimensions=64, walk_length=walk_length, num_walks=num_walks, p=0.5, q=2.0, window_size=5, workers=2, seed=42)
        bfs_embs = n2v_bfs.fit(G_train)
        bfs_auc, _ = predictor.train_and_evaluate(bfs_embs, 64, train_pos, train_neg, test_pos, test_neg)
        table_data[dname]["Node2Vec (BFS-like)"] = bfs_auc

        # C. Node2Vec (DFS-like)
        print(f"  🔴 训练 Node2Vec (DFS-like: p=2.0, q=0.5)...")
        n2v_dfs = Node2Vec(dimensions=64, walk_length=walk_length, num_walks=num_walks, p=2.0, q=0.5, window_size=5, workers=2, seed=42)
        dfs_embs = n2v_dfs.fit(G_train)
        dfs_auc, _ = predictor.train_and_evaluate(dfs_embs, 64, train_pos, train_neg, test_pos, test_neg)
        table_data[dname]["Node2Vec (DFS-like)"] = dfs_auc

        # ----------------- 3. Ours (Jaccard Weighted Walk) -----------------
        print(f"  🚀 训练我们的 Jaccard 加权游走 (创新算法: beta=0.2)...")
        weighted_dw = JaccardWeightedDeepWalk(
            dimensions=64, walk_length=walk_length, num_walks=num_walks,
            beta=0.2, window_size=5, workers=2, seed=42
        )
        weighted_embs = weighted_dw.fit(G_train)
        weighted_auc, _ = predictor.train_and_evaluate(weighted_embs, 64, train_pos, train_neg, test_pos, test_neg)
        table_data[dname]["Ours (Weighted Walk)"] = weighted_auc

    # 打印最终对比表格
    print("\n" + "=" * 80)
    print("📊 论文汇总表：三数据集链路预测 AUC-ROC 对比结果")
    print("=" * 80)
    print(f"  {'算法/方法':28s} | {'Cora':^12s} | {'Citeseer':^12s} | {'Facebook':^12s}")
    print(f"  {'-'*28}-|-{'-'*12}-|-{'-'*12}-|-{'-'*12}")
    for m in methods:
        c_auc = table_data["cora"][m]
        t_auc = table_data["citeseer"][m]
        f_auc = table_data["facebook"][m]
        
        # 将最优的值标粗 (这里用星号模拟，便于阅读)
        # 找出当前数据集的最大值
        max_c = max(table_data["cora"][x] for x in methods)
        max_t = max(table_data["citeseer"][x] for x in methods)
        max_f = max(table_data["facebook"][x] for x in methods)
        
        c_str = f"*{c_auc:.4f}*" if c_auc == max_c else f" {c_auc:.4f} "
        t_str = f"*{t_auc:.4f}*" if t_auc == max_t else f" {t_auc:.4f} "
        f_str = f"*{f_auc:.4f}*" if f_auc == max_f else f" {f_auc:.4f} "
        
        print(f"  {m:28s} | {c_str:^12s} | {t_str:^12s} | {f_str:^12s}")
    print("=" * 80)
    print("  (注：*号包围的数据为各数据集上的最佳预测表现)")

    # 写入最终 txt 报告
    report_path = "results/tables/final_cross_dataset_results.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("三数据集链路预测实验结果大表 (AUC-ROC)\n")
        f.write("=" * 75 + "\n")
        f.write(f"{'算法/方法':28s} | {'Cora':^12s} | {'Citeseer':^12s} | {'Facebook':^12s}\n")
        f.write("-" * 75 + "\n")
        for m in methods:
            c_auc = table_data["cora"][m]
            t_auc = table_data["citeseer"][m]
            f_auc = table_data["facebook"][m]
            f.write(f"{m:28s} | {c_auc:12.4f} | {t_auc:12.4f} | {f_auc:12.4f}\n")
        f.write("=" * 75 + "\n")
    print(f"\n📝 最终数据表已写入报告: {report_path}")


def run_ablation_study():
    """
    【实验二】消融实验与超参敏感性分析 (以 Cora 数据集为例)
    """
    print("\n" + "=" * 65)
    print("🚀 正在启动：【实验二】创新算法消融实验 (Cora)")
    print("=" * 65)

    G, _, _ = load_cora()
    G_train, test_pos, test_neg = split_edges(G, test_ratio=0.2, seed=42)
    train_pos = list(G_train.edges())
    non_edges = list(nx.non_edges(G_train))
    rng = np.random.RandomState(42)
    rng.shuffle(non_edges)
    train_neg = non_edges[:len(train_pos)]
    
    predictor = LinkPredictor(operator="hadamard")

    # 1. 探索嵌入维度 Dimensions 的影响
    dims = [16, 32, 64, 128]
    dim_aucs = []
    print("\n📐 消融 A: 节点表示维度 (Dimensions) 的敏感性分析...")
    for d in dims:
        weighted_dw = JaccardWeightedDeepWalk(
            dimensions=d, walk_length=40, num_walks=10,
            beta=0.2, window_size=5, workers=2, seed=42
        )
        embs = weighted_dw.fit(G_train)
        auc, _ = predictor.train_and_evaluate(embs, d, train_pos, train_neg, test_pos, test_neg)
        dim_aucs.append(auc)
        print(f"     Dimensions = {d:3d} | AUC = {auc:.4f}")

    # 2. 探索随机游走长度 Walk Length 的影响
    lengths = [10, 20, 40, 80]
    len_aucs = []
    print("\n📐 消融 B: 随机游走路径长度 (Walk Length) 的敏感性分析...")
    for l in lengths:
        weighted_dw = JaccardWeightedDeepWalk(
            dimensions=64, walk_length=l, num_walks=10,
            beta=0.2, window_size=5, workers=2, seed=42
        )
        embs = weighted_dw.fit(G_train)
        auc, _ = predictor.train_and_evaluate(embs, 64, train_pos, train_neg, test_pos, test_neg)
        len_aucs.append(auc)
        print(f"     Walk Length = {l:2d} | AUC = {auc:.4f}")

    # 3. 绘制学术风折线趋势图
    os.makedirs("results/figures", exist_ok=True)

    # 3a. 维度影响折线图
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(dims, dim_aucs, marker='s', color='#3b82f6', linewidth=2.5, markersize=8, label="Ours (beta=0.2)")
    ax.set_xlabel("Node Embedding Dimensions (d)", fontsize=12)
    ax.set_ylabel("AUC-ROC Score", fontsize=12)
    ax.set_title("Ablation Study: Effect of Embedding Dimensions\n(Cora Dataset)", fontsize=13, fontweight="bold")
    ax.set_xticks(dims)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    dim_path = "results/figures/ablation_dimensions.png"
    plt.tight_layout()
    plt.savefig(dim_path, dpi=150)
    print(f"\n📈 维度消融折线图已保存: {dim_path}")

    # 3b. 游走长度影响折线图
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(lengths, len_aucs, marker='^', color='#ef4444', linewidth=2.5, markersize=8, label="Ours (beta=0.2)")
    ax.set_xlabel("Random Walk Length (l)", fontsize=12)
    ax.set_ylabel("AUC-ROC Score", fontsize=12)
    ax.set_title("Ablation Study: Effect of Random Walk Length\n(Cora Dataset)", fontsize=13, fontweight="bold")
    ax.set_xticks(lengths)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    len_path = "results/figures/ablation_walk_length.png"
    plt.tight_layout()
    plt.savefig(len_path, dpi=150)
    print(f"📈 路径长度消融折线图已保存: {len_path}")

    # 写入消融实验报告
    ablation_report_path = "results/tables/ablation_study_results.txt"
    with open(ablation_report_path, "w") as f:
        f.write("我们的算法 (Jaccard Weighted DeepWalk) 消融实验结果 (Cora)\n")
        f.write("=" * 60 + "\n")
        f.write("1. 嵌入维度 Dimensions (d) 影响测试:\n")
        for d, a in zip(dims, dim_aucs):
            f.write(f"   - d = {d:<3d} : AUC = {a:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write("2. 游走长度 Walk Length (l) 影响测试:\n")
        for l, a in zip(lengths, len_aucs):
            f.write(f"   - l = {l:<2d}  : AUC = {a:.4f}\n")
        f.write("=" * 60 + "\n")
    print(f"📝 消融实验数据已写入报告: {ablation_report_path}")


def main():
    print("=" * 70)
    print("🎓 Week 7-8: 毕业论文终极汇总实验大联跑 (Datasets & Ablation)")
    print("=" * 70)

    # 1. 运行多数据集对比大表
    run_cross_dataset_experiments()

    # 2. 运行参数敏感性与消融实验
    run_ablation_study()

    print(f"\n{'='*70}")
    print("🎉 Week 7-8 终极数据图表大联跑已成功全部跑通！")
    print("   恭喜你！毕业论文所需的所有关键实验数据、折线图、结果对比表都已就绪。")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
