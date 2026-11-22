# 答辩与课程材料

**2026-11-23 中期答辩：从[总首页的上台口径](../README.md)开始。** 下列文件是已生成的阶段稿。实验结果还要复核，盲审按学院通知是 2027 年 3 月，不按旧页的 4 月安排。

| 材料 | 用法 |
|---|---|
| [中期放映 HTML](中期答辩_刘新宇.html) | 下载后离线打开；方向键翻页、`F` 全屏、`P` 打印。 |
| [中期放映 PDF](中期答辩_刘新宇.pdf) | 12 页横屏，作为投屏备份。 |
| [逐字稿](中期答辩逐字稿.md) / [问答准备](中期答辩问答准备.md) | 和放映稿对页用；结论与日期先按[总首页](../README.md)。 |
| [中期考核评价表](中期考核评价表_刘新宇_20261104.docx) | 现有填写版；确认并更新定稿后，由学生和导师签字。 |
| [CSIT998 模块 4](csit998/4_刘新宇.docx) | 课程作业，与中期答辩分开。 |

## 从源码重建

只想放映不用重建。下列出图脚本读取仓库中的结果文件，不会重跑实验；**复核实验后的更新顺序见[总首页](../README.md)**。

```bash
cd link-prediction                 # 从仓库根目录执行
uv run python build_figures.py     # 更新 deliverables/figures/ 下的矢量图
uv run python build_m4_figures.py  # 更新 CSIT998 图
cd ../deliverables
python3 build_deck.py              # deck_src.html + figures/*.svg → 单文件 HTML
```

打开 HTML 后按 `P`，可另存 12 页横版 PDF。`fill_midterm_form.py` 读取同目录的空白 `.docx` 模板，依赖 `python-docx`；其中写死的阶段数据要先更新，再用它生成新的评价表。
