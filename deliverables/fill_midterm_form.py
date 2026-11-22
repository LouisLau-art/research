"""
fill_midterm_form.py — 填写伍伦贡联合研究院《研究生学位论文中期考核评价表》
══════════════════════════════════════════════════════════════════════
基于学院官方 .doc 模板（docs/school/伍伦贡联合研究院研究生学位论文中期考核评价表.doc，
已由 LibreOffice 转为 docx），保留原有版式、字体与表格结构，仅填入内容。

输出: deliverables/中期考核评价表_刘新宇_20261104.docx
"""
import os
import re

from docx import Document
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, "伍伦贡联合研究院研究生学位论文中期考核评价表.docx")
OUT = os.path.join(HERE, "中期考核评价表_刘新宇_20261104.docx")

# ── 封面信息 ────────────────────────────────────────────────────
COVER = {
    "论  文  题  目": "基于网络表示学习与节点结构相似度的社交网络链路预测研究",
    "研 究 生 姓 名": "刘新宇",
    "学          号": "2025124630",
    "导  师  姓  名": "崔建群 教授",
    "专          业": "计算机科学与技术",
    "研  究  方  向": "社交网络分析 / 网络表示学习",
    "入  学  时  间": "2025 年 9 月",
    "毕  业  时  间": "2027 年 6 月",
}

# ── 正文内容（每项为若干段） ──────────────────────────────────────
BODY = {
    "是否按开题报告预定的内容及论文计划进度进行": [
        "按开题计划推进，进度与开题报告一致，未滞后。开题答辩（2026 年 7 月 7 日）后，"
        "Phase 1（文献综述、开发环境搭建、三个数据集下载与预处理）与 Phase 2（基线算法实现与"
        "三数据集初步实验）均已完成；目前处于 Phase 3（创新算法设计与超参数分析，开题计划为"
        "第 5—6 个月）。",

        "有两处调整需说明，均不涉及研究内容与目标的变更：",

        "（1）实现顺序调整。开题方案中的加权随机游走为「Jaccard 系数 × Node2Vec 偏置 α(t,x)」"
        "的联合形式。为隔离变量、便于归因，前期先实现了不含 p/q 偏置的 Jaccard-DeepWalk 版本；"
        "按开题公式叠加 Node2Vec 偏置的联合版本已于 2026 年 11 月补齐并在统一划分下完成对比。",

        "（2）研究重点调整（有实证依据）。补齐开题承诺的「10 次重复实验 + 报告标准差」后发现，"
        "单次随机划分给出的排名不可靠；进一步追查原因，确认问题出在通用评估协议本身"
        "（随机负采样低估度数、向量模长被当作判边依据），而非算法机制失效。"
        "因此本阶段把部分工作量从「继续调参」转向「评估协议纠偏与边界条件分析」，"
        "论文贡献重心相应从「提出一个更强的加权方法」调整为"
        "「结构先验注入随机游走的效果边界 + 链路预测评估协议的可靠性」——此调整已在汇报中说明。",
    ],

    "已完成的研究内容及取得的成果": [
        "1. 文献调研：精读并梳理链路预测与网络表示学习方向核心文献，包括 DeepWalk、Node2Vec、"
        "LINE、Liben-Nowell 与 Kleinberg 的经典链路预测研究、Adamic-Adar 指标、Zhou 等的资源"
        "分配指标以及 Kumar 等的综述，据此明确「结构相似度与表示学习在采样阶段深度融合」"
        "这一研究切入点。",

        "2. 实验平台：搭建 Python 全流程可复现实验管道（networkx + gensim + scikit-learn），"
        "涵盖数据加载、边的随机划分（藏起 20% 作正样本）、负样本采样、加权随机游走、"
        "Skip-gram 节点嵌入、Hadamard 边特征、逻辑回归分类与 AUC-ROC 评价；"
        "并实现断点续跑与结果落盘，代码托管于私有仓库 link-prediction。",

        "3. 基线复现：在 Cora、Citeseer、Facebook 三个公开数据集上复现 7 种方法"
        "（共同邻居 CN、Jaccard 系数、Adamic-Adar、DeepWalk、Node2Vec-BFS、Node2Vec-DFS、"
        "本文方法），全部方法在同一套边划分、同一套负样本、同一个分类器下比较。",

        "4. 创新算法实现：完成结构相似度加权随机游走，转移概率为 "
        "P(v→x) ∝ α(t,x)·w(v,x)·( s(v,x)+β )^γ，其中 s 可取 Jaccard / 资源分配(RA) / Adamic-Adar，"
        "γ 为锐化指数；实现相似度缓存以降低采样开销；参数均可插拔，便于系统做对照实验。"
        "完成超参分析：嵌入维度 d 在 64 之后收益饱和（0.798），游走长度 l 在 10—80 之间"
        "影响小于 0.006，说明瓶颈不在采样规模。",

        "5. 实验方法论升级：将「单次随机划分」升级为「10 次随机划分 + 均值±标准差 + 配对 t 检验 "
        "与 Wilcoxon 符号秩检验」，与开题承诺对齐。",

        "6. 阶段性实证成果（均可复现，是论文实验与讨论章的核心素材）：",

        "（1）单次划分不可靠：同一模型仅更换随机划分，AUC 在 0.775—0.823 间摆动"
        "（标准差 0.009—0.013），大于方法间差距（0.001—0.006）。以 seed=42 单次实验会得到与 10 次"
        "平均相反的排名（Cora 上单次本文方法 0.8143 高于 DeepWalk 0.8075，10 次平均为 0.7910 对 0.7974）。",

        "（2）评估协议缺陷一（随机负采样低估度数）：真实边天然更可能连接高度节点。"
        "Cora 上被藏起的正边平均「度积」为 76.0，随机采的负边仅 17.0（差 4.5 倍）；Citeseer 为 54.8 与 7.8（差 7 倍）。"
        "仅用「度积」这一个标量做逻辑回归，Cora 即可取得 0.631 的 AUC；"
        "改用度数分层匹配的负样本后，该基线降至 0.511（等同随机猜测）。"
        "说明该协议会白送与算法质量无关的 AUC。",

        "（3）评估协议缺陷二（向量模长捷径）：Hadamard 边特征的幅值随节点频次（约等于度数）增长，"
        "分类器可直接以「模长」判边。典型证据：将锐化指数 γ 由 1 提高到 8，Citeseer 上 AUC 由 "
        "0.654 升至 0.747，看似大幅领先；但把向量归一化为单位长度后立即回落至 0.656，"
        "说明提升来自模长而非结构信息。",

        "（4）修正协议下的真实排名（B 协议 = 度数匹配负样本 + 单位归一化向量，同样 10 个划分）："
        "① 表示学习显著优于启发式——Cora 上 DeepWalk 0.8001 对 CN 0.6986（Δ=+0.1015，10/10 胜，"
        "p<0.001），Citeseer 上 0.6973 对 0.6514（Δ=+0.0459，10/10 胜，p<0.001）；"
        "原「稀疏图上嵌入不如简单启发式」的结论系协议缺陷所致，不成立。"
        "② 本文方法与 DeepWalk 无统计差异——Cora 0.8004 对 0.8001（p=0.85），"
        "Citeseer 0.6963 对 0.6973（p=0.43）；RA、AA 加权同样无差异（p=0.41—0.85）。"
        "③ 方法间差异同时被压缩，标准差 Cora 由 0.0130 降至 0.0104、Citeseer 由 0.0091 降至 0.0072，"
        "说明原协议下部分「方法优劣」表象来自度数噪声。",
    ],

    "目前存在的或预期可能出现的问题，拟采用的解决方案": [
        "问题 1：结构相似度加权方案在 Cora、Citeseer 上无可测量增益。"
        "证据：A/B 两套协议下，Jaccard、RA、AA 三种加权与 DeepWalk 均无统计差异；"
        "换融合位置同样无增益——混合预测器（结构相似度作为显式特征与嵌入拼接）已完成 8 组对照，"
        "最大提升仅 0.0008，个别组统计显著但实际意义为零，说明嵌入与结构信息高度冗余。"
        "解决方案：论文主线已定为「结构先验注入的边界条件 + 评估协议纠偏」（负结果 + 方法学发现），"
        "不再寻找「更强加权方案」；后续把该结论在更多数据集上复核。",

        "问题 2：评估协议缺陷尚未系统化、成套化。目前只量化了两类缺陷、两个数据集。"
        "解决方案：把修正协议（度数匹配负样本 + 单位归一化 + 多种子配对检验）整理为完整评测脚本，"
        "在 Facebook、DBLP 上复现；补充 Hits@K、MRR 等排序敏感指标，避免单一 AUC 掩盖差异。",

        "问题 3：数据集覆盖不足。Facebook 上所有方法 AUC 均在 0.986—0.991，方法间差异小于随机波动，"
        "失去区分度；仅两个有效数据集不足以支撑「边界条件」类结论。"
        "解决方案：补入 DBLP 合作网（中等密度）与 Twitter ego 网络，覆盖由稀疏到稠密的密度区间。",

        "问题 4：网络密度、聚类系数、社区结构对增益影响的系统分析尚未开展（对应开题 Phase 4 前半），"
        "预计 2026 年 12 月完成，不影响论文整体进度。",
    ],

    "下一步的工作计划和研究内容": [
        "2026 年 11 月：方法定稿。混合预测器已验证（无增益，采样与特征两级融合均已排除），"
        "余下工作在四个数据集上以双协议复核，论文主线锁定为「负结果 + 方法学纠偏」（对应开题 Phase 3）。",

        "2026 年 12 月：完成完整实验矩阵（4 个数据集 × 10 种方法 × 10 次随机划分 × 2 套协议），"
        "全部报告均值±标准差与配对显著性；完成网络密度、聚类系数、社区结构对增益的影响分析"
        "（对应开题 Phase 4 前半）。",

        "2027 年 1—2 月：撰写论文主体五章（引言、相关工作、方法、实验、讨论），英文撰写，"
        "采用 JI 的 LaTeX 模板（对应开题 Phase 4 后半）。",

        "2027 年 3 月：完成论文初稿并提交导师审阅，同期提交《学位论文初稿审核意见表》"
        "（对应开题 Phase 5）。",

        "2027 年 4 月：按导师意见修订定稿，完成查重、盲评材料准备与答辩演练（对应开题 Phase 6）。",
    ],

    "学生自评": [
        "本人按开题计划完成了前两个阶段的全部工作：实验平台搭建完毕且全部可复现，三个数据集上的"
        "7 种方法均已复现，创新算法已按开题公式实现并完成超参数分析。",

        "在此基础上主动把实验从单次划分升级为 10 次重复实验加配对显著性检验，并进一步追查"
        "「为什么提升不成立」，最终定位到通用评估协议的两类缺陷（随机负采样低估度数、"
        "向量模长被当作判边依据），并自行设计修正协议予以验证。这一过程使前期结论被否定，"
        "但也使结论更可靠：修正后既确认了表示学习相对启发式的真实优势（+0.050—+0.099），"
        "也确认了本文加权方案尚无可测量增益。",

        "当前主要问题是论文贡献表述需要重构；混合预测器已于 11 月初完成验证（无增益），"
        "「换融合位置」的出路已由实验排除，主线随之确定，不存在进度滞后。"
        "通过本阶段工作，文献阅读、编程实现、实验设计与结果批判性分析能力均得到锻炼，"
        "在剩余 5 个月内具备按期完成学位论文的条件。",
    ],
}


def set_cover(doc):
    """
    封面填写：保留「前导空格 + 标签」，删除标签后的空白 run，
    再用一个继承下划线格式的 run 追加「　+ 值」。
    """
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    import copy as _copy

    for p in doc.paragraphs:
        norm = re.sub(r"\s+", "", p.text)
        for label, value in COVER.items():
            if not norm.startswith(re.sub(r"\s+", "", label)):
                continue
            runs = list(p.runs)
            label_end = None
            for i, r in enumerate(runs):
                if r.text.strip():
                    label_end = i
            if label_end is None:
                break
            tail = [r for r in runs[label_end + 1:]]
            proto = tail[0]._element if tail else runs[label_end]._element
            for r in tail:
                r._element.getparent().remove(r._element)
            new_el = _copy.deepcopy(proto)
            for t in new_el.findall(qn("w:t")):
                new_el.remove(t)
            t = OxmlElement("w:t")
            t.text = "\u3000" + value
            t.set(qn("xml:space"), "preserve")
            new_el.append(t)
            runs[label_end]._element.addnext(new_el)
            break


def fill_cell(cell, paragraphs):
    """cell 的第 0 段是标题，其余为空段；用它们承载内容。"""
    header = cell.paragraphs[0]
    proto_run = header.runs[0] if header.runs else None

    blanks = cell.paragraphs[1:]
    for idx, text in enumerate(paragraphs):
        if idx < len(blanks):
            p = blanks[idx]
        else:
            p = cell.add_paragraph()
            p._p.getparent().remove(p._p)
            cell._tc.append(p._p)
        for r in list(p.runs):
            r._element.getparent().remove(r._element)
        run = p.add_run(text)
        if proto_run is not None:
            run.font.name = proto_run.font.name
            run.font.size = proto_run.font.size
        # 「（1）（2）」等编号段不缩进，正文段缩进两字
        indent = Pt(0) if re.match(r"^（\d）", text) else Pt(21)
        p.paragraph_format.first_line_indent = indent
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15


def main():
    doc = Document(TPL)
    set_cover(doc)

    table = doc.tables[0]
    # 第 0 行：学位论文题目
    r0 = table.rows[0].cells[0]
    p0 = r0.paragraphs[0]
    run = p0.add_run(COVER["论  文  题  目"])
    if p0.runs:
        run.font.name = p0.runs[0].font.name
        run.font.size = p0.runs[0].font.size
        run.font.bold = False

    filled = 0
    for row in table.rows:
        cell = row.cells[0]
        head = cell.paragraphs[0].text
        for key, paras in BODY.items():
            if head.startswith(key):
                fill_cell(cell, paras)
                filled += 1
                break
    print(f"✔ 已填写 {filled} 个正文区块（应为 5）")

    for p in doc.paragraphs:
        if "二〇二" in p.text:
            for r in p.runs:
                if "二〇二" in r.text:
                    r.text = "二〇二六年十一月"
            break

    doc.save(OUT)
    print(f"✅ 已保存 {OUT}")


if __name__ == "__main__":
    main()
