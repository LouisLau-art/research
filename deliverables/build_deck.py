"""
build_deck.py — 把 SVG 图表内联进 HTML 演示页
══════════════════════════════════════════════════════════════════════
用法: python build_deck.py
输入: deliverables/deck_src.html + deliverables/figures/*.svg
输出: deliverables/中期答辩_刘新宇.html（自包含，单个文件即可放映/打印）
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "deck_src.html")
FIGDIR = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "中期答辩_刘新宇.html")


def strip_svg_header(svg: str) -> str:
    """去掉 XML 声明与 DOCTYPE，保留 <svg> 本体。"""
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
    return svg.strip()


def main():
    html = open(SRC, encoding="utf-8").read()

    def repl(m):
        name = m.group(1)
        path = os.path.join(FIGDIR, name)
        if not os.path.exists(path):
            print(f"  ⚠ 缺少图 {name}")
            return f'<div style="color:#e11d48">[missing {name}]</div>'
        svg = strip_svg_header(open(path, encoding="utf-8").read())
        # 让 SVG 自适应容器
        svg = svg.replace("<svg ", '<svg preserveAspectRatio="xMidYMid meet" ', 1)
        print(f"  ✔ 内联 {name}  ({len(svg)//1024} KB)")
        return svg

    out = re.sub(r"<!--FIG:([^>]+?)-->", repl, html)
    open(OUT, "w", encoding="utf-8").write(out)
    print(f"\n✅ 生成 {OUT}  ({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
