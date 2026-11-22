"""
graph_utils.py - 图数据加载和处理工具
========================================
Week 1 核心代码：下载数据集、构建图、基本统计

教学重点：
- 理解图的节点(node)和边(edge)
- 理解邻接表(adjacency list)和邻接矩阵(adjacency matrix)
- 理解度(degree)的含义
"""

import os
import urllib.request
import tarfile
import zipfile
import gzip
import networkx as nx
import numpy as np


# ============================================================
# 数据集下载
# ============================================================

DATASET_URLS = {
    "cora": "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    "citeseer": "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz",
    "facebook": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
}

def download_dataset(name: str, data_dir: str = "data") -> str:
    """
    下载并解压数据集。

    参数:
        name: 数据集名称 ("cora", "citeseer")
        data_dir: 数据存放的根目录

    返回:
        解压后的数据集目录路径
    """
    target_dir = os.path.join(data_dir, name)
    os.makedirs(target_dir, exist_ok=True)

    # 检查是否已下载
    if name == "cora" and os.path.exists(os.path.join(target_dir, "cora.cites")):
        print(f"✅ {name} 数据集已存在，跳过下载")
        return target_dir
    if name == "citeseer" and os.path.exists(os.path.join(target_dir, "citeseer.cites")):
        print(f"✅ {name} 数据集已存在，跳过下载")
        return target_dir
    if name == "facebook" and os.path.exists(os.path.join(target_dir, "facebook_combined.txt")):
        print(f"✅ {name} 数据集已存在，跳过下载")
        return target_dir

    if name not in DATASET_URLS:
        raise ValueError(f"未知数据集: {name}，支持: {list(DATASET_URLS.keys())}")

    url = DATASET_URLS[name]
    
    if name == "facebook":
        archive_path = os.path.join(target_dir, "facebook_combined.txt.gz")
        dest_path = os.path.join(target_dir, "facebook_combined.txt")
        print(f"📥 正在下载 {name} 数据集...")
        print(f"   URL: {url}")
        urllib.request.urlretrieve(url, archive_path)
        print(f"   下载完成: {archive_path}")
        print(f"📦 正在使用 gzip 解压...")
        with gzip.open(archive_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(archive_path)
        print(f"✅ {name} 数据集已就绪: {target_dir}")
        return target_dir

    archive_path = os.path.join(data_dir, f"{name}.tgz")

    print(f"📥 正在下载 {name} 数据集...")
    print(f"   URL: {url}")
    urllib.request.urlretrieve(url, archive_path)
    print(f"   下载完成: {archive_path}")

    print(f"📦 正在解压...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")

    # 清理压缩文件
    os.remove(archive_path)
    print(f"✅ {name} 数据集已就绪: {target_dir}")
    return target_dir



# ============================================================
# 图构建
# ============================================================

def load_cora(data_dir: str = "data/cora") -> tuple[nx.Graph, dict, dict]:
    """
    加载 Cora 引文网络数据集。

    Cora 数据集说明:
    ────────────────
    - 这是一个学术论文的引用网络
    - 每个节点 = 一篇论文
    - 每条边 = 一篇论文引用了另一篇
    - 共 2708 篇论文，分为 7 个类别
    - 每篇论文用 1433 维的词袋向量描述（特征）

    文件格式:
    - cora.content: 每行 = <论文ID> <word_1> <word_2> ... <word_1433> <类别>
    - cora.cites:   每行 = <被引论文ID> <引用论文ID>

    返回:
        G: networkx 无向图
        features: {node_id: numpy array} 节点特征
        labels: {node_id: str} 节点类别
    """
    content_file = os.path.join(data_dir, "cora.content")
    cites_file = os.path.join(data_dir, "cora.cites")

    if not os.path.exists(content_file):
        raise FileNotFoundError(
            f"找不到 {content_file}，请先运行 download_dataset('cora')"
        )

    features = {}
    labels = {}
    node_ids = set()

    # 读取节点信息
    print("📄 读取 cora.content（节点特征和标签）...")
    with open(content_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            node_id = int(parts[0])
            feature_vec = np.array([int(x) for x in parts[1:-1]], dtype=np.float32)
            label = parts[-1]

            node_ids.add(node_id)
            features[node_id] = feature_vec
            labels[node_id] = label

    # 构建图
    G = nx.Graph()
    # 必须按确定顺序插入：node_ids 是 set，元素为字符串时迭代顺序随
    # PYTHONHASHSEED 变化，会让 list(G.nodes())/list(G.edges()) 的顺序
    # 跨进程不同，进而让「同一 seed 的边划分」在两次运行里抽到不同的边，
    # 表现为「同一配置 AUC 抖动 ±0.01」，把方法间 0.004 的差距淹掉。
    G.add_nodes_from(sorted(node_ids, key=str))

    # 读取边（引用关系）
    print("📄 读取 cora.cites（引用关系 = 边）...")
    edge_count = 0
    skipped = 0
    with open(cites_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            # 只添加两端节点都存在的边
            if u in node_ids and v in node_ids:
                G.add_edge(u, v)
                edge_count += 1
            else:
                skipped += 1

    print(f"✅ 图构建完成（跳过了 {skipped} 条端点不存在的边）")
    return G, features, labels


def load_citeseer(data_dir: str = "data/citeseer") -> tuple[nx.Graph, dict, dict]:
    """
    加载 Citeseer 引文网络数据集。

    Citeseer 说明:
    ─────────────
    - 类似 Cora 的学术引用网络
    - 3312 篇论文，6 个类别
    - 每篇论文用 3703 维特征描述
    """
    content_file = os.path.join(data_dir, "citeseer.content")
    cites_file = os.path.join(data_dir, "citeseer.cites")

    if not os.path.exists(content_file):
        raise FileNotFoundError(
            f"找不到 {content_file}，请先运行 download_dataset('citeseer')"
        )

    features = {}
    labels = {}
    node_ids = set()

    print("📄 读取 citeseer.content...")
    with open(content_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            node_id = parts[0]  # Citeseer 的 ID 是字符串
            feature_vec = np.array([int(x) for x in parts[1:-1]], dtype=np.float32)
            label = parts[-1]

            node_ids.add(node_id)
            features[node_id] = feature_vec
            labels[node_id] = label

    G = nx.Graph()
    # 必须按确定顺序插入：node_ids 是 set，元素为字符串时迭代顺序随
    # PYTHONHASHSEED 变化，会让 list(G.nodes())/list(G.edges()) 的顺序
    # 跨进程不同，进而让「同一 seed 的边划分」在两次运行里抽到不同的边，
    # 表现为「同一配置 AUC 抖动 ±0.01」，把方法间 0.004 的差距淹掉。
    G.add_nodes_from(sorted(node_ids, key=str))

    print("📄 读取 citeseer.cites...")
    edge_count = 0
    skipped = 0
    with open(cites_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            u, v = parts[0], parts[1]
            if u in node_ids and v in node_ids:
                G.add_edge(u, v)
                edge_count += 1
            else:
                skipped += 1

    print(f"✅ 图构建完成（跳过了 {skipped} 条端点不存在的边）")
    return G, features, labels


def load_facebook(data_dir: str = "data/facebook") -> tuple[nx.Graph, None, None]:
    """
    加载 Facebook 社交网络数据集。

    Facebook 数据集说明:
    ──────────────────
    - 这是一个真实的社交网络图（来自 Stanford SNAP 平台）
    - 包含 4,039 个用户节点，88,234 条朋友连接边
    - 本数据集纯为拓扑结构，不包含节点的文本分类标签或描述特征。
    """
    edges_file = os.path.join(data_dir, "facebook_combined.txt")
    if not os.path.exists(edges_file):
        raise FileNotFoundError(
            f"找不到 {edges_file}，请先运行 download_dataset('facebook')"
        )

    print("📄 读取 facebook_combined.txt 并构建图...")
    G = nx.read_edgelist(edges_file, nodetype=int)
    print(f"✅ Facebook 图构建完成！节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    return G, None, None



# ============================================================
# 图统计分析
# ============================================================

def print_graph_stats(G: nx.Graph, name: str = "Graph"):
    """
    打印图的基本统计信息。

    这些统计量帮助你理解这个网络的"形状"：
    - 节点数 = 网络中有多少个实体
    - 边数 = 有多少对实体之间有关系
    - 平均度 = 平均每个人有几个朋友
    - 密度 = 实际边数 / 理论最大边数（密度越低，图越稀疏）
    - 连通分量数 = 图被分成了几个"孤岛"
    - 最大连通分量 = 最大的那个"孤岛"有多少节点
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_deg = np.mean(degrees)
    density = nx.density(G)

    # 连通分量
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)

    print(f"\n{'='*55}")
    print(f"📊 {name} 网络统计信息")
    print(f"{'='*55}")
    print(f"  节点数 (|V|):           {n:,}")
    print(f"  边数   (|E|):           {m:,}")
    print(f"  平均度 (avg degree):    {avg_deg:.2f}")
    print(f"  最大度 (max degree):    {max(degrees)}")
    print(f"  最小度 (min degree):    {min(degrees)}")
    print(f"  密度   (density):       {density:.6f}")
    print(f"  连通分量数:             {len(components)}")
    print(f"  最大连通分量节点数:     {len(largest_cc)}")
    print(f"{'='*55}")

    return {
        "nodes": n,
        "edges": m,
        "avg_degree": avg_deg,
        "max_degree": max(degrees),
        "density": density,
        "num_components": len(components),
        "largest_cc_size": len(largest_cc),
    }


def get_degree_distribution(G: nx.Graph) -> tuple[list, list]:
    """
    计算度分布。

    度分布告诉你：
    - 有多少节点有 1 个邻居？
    - 有多少节点有 2 个邻居？
    - 有多少节点有 100 个邻居？

    社交网络通常呈现"幂律分布"（Power Law）：
    大部分人只有几个朋友，极少数人有成百上千的朋友（"大V"/"意见领袖"）
    """
    degree_count = {}
    for _, d in G.degree():
        degree_count[d] = degree_count.get(d, 0) + 1

    degrees = sorted(degree_count.keys())
    counts = [degree_count[d] for d in degrees]
    return degrees, counts


def all_non_edges(G: nx.Graph) -> list:
    """
    返回图中所有不存在的节点对，顺序完全确定（不依赖 PYTHONHASHSEED）。

    为什么不直接用 networkx 的 nx.non_edges：
        nx 3.x 的实现是 `nodes = set(graph); while nodes: u = nodes.pop()`，
        节点为字符串时 set 的迭代顺序随 PYTHONHASHSEED 变化，同一张图、
        同一个 seed 在两次进程运行里会得到不同顺序的 non_edges，
        负样本随之一致性丢失——表现为「同一配置的 AUC 跨进程抖动 ±0.01」，
        足以淹没方法之间 0.004 量级的差距。
    """
    nodes = sorted(G.nodes(), key=str)
    out = []
    for i, u in enumerate(nodes):
        nu = G[u]
        for v in nodes[i + 1:]:
            if v not in nu:
                out.append((u, v))
    return out


# ============================================================
# 边分割（链路预测实验用）
# ============================================================

def split_edges(G: nx.Graph, test_ratio: float = 0.2, seed: int = 42):
    """
    将图的边分割为训练集和测试集。

    链路预测实验的标准流程：
    ────────────────────────
    1. 从图中随机"藏起"一部分边（test_ratio，默认 20%）
    2. 在剩下的边上训练模型
    3. 让模型预测被藏起的边 → 看它能不能猜对

    正样本 = 真实存在的边（应该预测为 1）
    负样本 = 不存在的边（应该预测为 0）

    参数:
        G: 原始图
        test_ratio: 测试集占比
        seed: 随机种子（保证实验可复现）

    返回:
        G_train: 训练图（去掉了测试边）
        test_pos_edges: 被藏起的边（正样本）
        test_neg_edges: 不存在的边（负样本）
    """
    rng = np.random.RandomState(seed)

    edges = list(G.edges())
    num_test = int(len(edges) * test_ratio)

    # 打乱后分割
    rng.shuffle(edges)
    test_pos_edges = edges[:num_test]
    train_edges = edges[num_test:]

    # 构建训练图
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)

    # 确保训练图中没有孤立节点导致的断连
    # （实际上我们允许，但要记录一下）
    isolated = list(nx.isolates(G_train))
    if isolated:
        print(f"⚠️  移除测试边后产生了 {len(isolated)} 个孤立节点")

    # 生成负样本（随机采样不存在的边；用确定性枚举，见 all_non_edges 注释）
    non_edges = all_non_edges(G_train)
    rng.shuffle(non_edges)
    test_neg_edges = non_edges[:num_test]

    print(f"\n📊 边分割结果:")
    print(f"  训练边数: {len(train_edges)}")
    print(f"  测试正样本: {len(test_pos_edges)}")
    print(f"  测试负样本: {len(test_neg_edges)}")

    return G_train, test_pos_edges, test_neg_edges
