"""
Week 3 实操脚本：Word2Vec 原理与 gensim 实操
════════════════════════════════════════════════
运行方式: uv run python week3_word2vec_explore.py

本脚本会带你完成两个任务：
【任务 1】在文本语料上训练 Word2Vec，直观感受“近义词”和“词向量加减法”。
【任务 2】在随机游走路径上训练 Word2Vec，获得图节点嵌入向量，并检验它们拓扑相似度。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from gensim.models import Word2Vec
import networkx as nx

from src.graph_utils import download_dataset, load_cora
from src.random_walk import generate_walks


def task1_text_embedding():
    """
    【任务 1】在简单的英文文本上训练 Word2Vec
    帮助建立对词向量在语义表示上的直觉。
    """
    print("\n" + "─" * 50)
    print("📝 任务 1: 文本词向量训练与语义探索")
    print("─" * 50)

    # 准备一份微型的模拟文本语料
    # 每行代表一个“句子”，每个元素是“词”
    sentences = [
        ["king", "is", "a", "royal", "man"],
        ["queen", "is", "a", "royal", "woman"],
        ["prince", "is", "a", "royal", "boy"],
        ["princess", "is", "a", "royal", "girl"],
        ["man", "loves", "beer", "and", "sports"],
        ["woman", "loves", "wine", "and", "fashion"],
        ["boy", "likes", "toy", "cars", "and", "games"],
        ["girl", "likes", "dolls", "and", "dress"],
        ["apple", "and", "banana", "are", "fruits", "to", "eat"],
        ["orange", "is", "a", "sweet", "fruit"],
        ["grapes", "are", "juicy", "fruits"],
    ]

    print("  1. 正在初始化并训练 Word2Vec 模型...")
    # sg=1 代表使用 Skip-gram 模式；sg=0 则为 CBOW
    # vector_size=16 维度设低一些方便观察
    # min_count=1 只要出现 1 次的词都计入词表
    # window=3 滑动窗口大小为 3
    # negative=5 每个正样本采样 5 个负样本
    model = Word2Vec(
        sentences,
        vector_size=16,
        window=3,
        min_count=1,
        sg=1,
        negative=5,
        workers=1,
        seed=42
    )

    print("  2. 词表中的单词数量:", len(model.wv))
    
    # 查找某个词的词向量
    word = "king"
    vector = model.wv[word]
    print(f"\n  3. 单词 '{word}' 的 {len(vector)} 维词嵌入向量为:\n     {vector}")

    # 4. 观察近义词/语义相似性
    print("\n  4. 寻找在语义上最接近的词 (余弦相似度):")
    for test_word in ["king", "fruit", "woman"]:
        similar_words = model.wv.most_similar(test_word, topn=3)
        print(f"     与 '{test_word}' 最相似的词:")
        for sim_word, score in similar_words:
            print(f"       👉 {sim_word:10s} (相似度: {score:.4f})")

    # 5. 著名的词向量加减法演示: king - man + woman = queen
    # 我们用相似度来验证这个方向
    print("\n  5. 词向量算术运算演示 (类比推理):")
    # king - man + woman 应该接近 queen
    result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
    print("     'king' - 'man' + 'woman' 的计算结果最接近:")
    for sim_word, score in result:
        print(f"       👉 {sim_word:10s} (相似度: {score:.4f})")
    
    print("\n  💡 任务 1 结论:")
    print("     我们没有告诉模型任何关于'性别'、'职业'或'水果'的规则，")
    print("     仅仅通过它们在句子中的共同出现，模型就自动学到了单词背后的语义结构！")


def task2_graph_embedding():
    """
    【任务 2】在 Cora 图随机游走路径上训练 Word2Vec (DeepWalk 的前两步)
    """
    print("\n" + "─" * 50)
    print("🕸️ 任务 2: 在图随机游走上训练 Word2Vec (节点嵌入)")
    print("─" * 50)

    # 1. 加载 Cora 并切分图
    print("  1. 加载 Cora 数据集...")
    G, _, labels = load_cora()
    
    # 2. 生成随机游走
    print("\n  2. 对 Cora 节点做均匀随机游走 (生成句子)...")
    # 为了演示速度，这里我们降低每次的游走数量和长度
    # 真实 DeepWalk 通常每个节点游走 40~80 次，长度为 40~80
    walks = generate_walks(G, num_walks=10, walk_length=40, seed=42)
    
    # 💡 关键：Word2Vec 需要接收字符串列表类型的“句子”
    # 我们把节点 ID (数字) 转换成字符串
    str_walks = [[str(node) for node in walk] for walk in walks]
    
    print(f"     生成的'句子'数: {len(str_walks)}")
    print(f"     示例句子 1: {str_walks[0][:15]}...")

    # 3. 训练 Word2Vec 得到节点嵌入
    print("\n  3. 用 Word2Vec 训练这些随机游走句子...")
    # vector_size=64: 每个节点学一个 64 维向量
    # sg=1: 必须用 Skip-gram 捕获远距离结构
    # window=5: 捕捉 5 步以内的跳跃关系
    model_dw = Word2Vec(
        str_walks,
        vector_size=64,
        window=5,
        min_count=0,
        sg=1,
        workers=2,
        seed=42
    )
    
    wv = model_dw.wv
    print(f"     成功为 {len(wv)} 个节点学习了 64 维向量！")

    # 4. 验证节点嵌入在空间上的拓扑聚集性
    print("\n  4. 验证：在向量空间里挨得近的节点，在图中是否也是邻居或同类？")
    
    # 找几个度数比较高的活跃节点
    degrees = dict(G.degree())
    active_nodes = sorted(degrees.items(), key=lambda x: -x[1])[:3]
    
    for node, deg in active_nodes:
        str_node = str(node)
        print(f"\n     中心节点: {node} (度 = {deg}, 类别 = {labels[node]})")
        
        # 寻找在 64 维向量空间中距离最近的前 5 个节点
        sim_nodes = wv.most_similar(str_node, topn=5)
        print(f"     向量空间最接近的 5 个节点:")
        print(f"       {'目标节点':>8s}  {'余弦相似':>8s}  {'有边直接相连?':>12s}  {'类别是否相同?':>10s}")
        print(f"       {'─'*46}")
        
        for sim_str, score in sim_nodes:
            sim_node = int(sim_str)
            has_edge = "✅ 是" if G.has_edge(node, sim_node) else "❌ 否"
            same_class = "✅ 同类" if labels[node] == labels[sim_node] else "❌ 异类"
            print(f"       {sim_node:8d}  {score:8.4f}  {has_edge:>12s}  {same_class:>10s}")
            
    print("\n  💡 任务 2 结论:")
    print("     太神奇了！仅仅通过随机游走产生的'邻接路线'，")
    print("     节点在 64 维空间学到的表示就能紧密聚集邻居和同类别节点。")
    print("     这就是为什么图表示学习可以通过计算向量点积来预测'是否有连接关系'！")


def main():
    print("=" * 60)
    print("🎓 Week 3: Word2Vec 原理与 gensim 实操体验")
    print("=" * 60)

    # 运行任务 1: 文本词向量
    task1_text_embedding()

    # 运行任务 2: 图节点向量
    task2_graph_embedding()

    print(f"\n{'='*60}")
    print("🎉 Week 3 实验完成！")
    print("   你现在已经完全具备了复现经典 DeepWalk 的知识和工具基础。")
    print("   Week 4：我们将把本周的图节点 Word2Vec 向量插入链路预测框架，")
    print("   实现你人生中第一个完整的 Graph Embedding 链路预测 Pipeline！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
