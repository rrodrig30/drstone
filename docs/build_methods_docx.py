"""Render docs/01_DrStone_Methods.md to a .docx with embedded figures.

Handles the subset of Markdown this writeup uses: #/##/### headings, **bold**
inline runs, > blockquotes, and [[FIGURE: name.png]] placeholders (embedded from
docs/figures/). Run:  python docs/build_methods_docx.py
"""

from __future__ import annotations

import os
import re

from docx import Document
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "01_DrStone_Methods.md")
DOCX = os.path.join(HERE, "01_DrStone_Methods.docx")
FIGS = os.path.join(HERE, "figures")

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_FIG = re.compile(r"\[\[FIGURE:\s*(.+?)\s*\]\]")


def add_runs(par, text):
    """Add text to a paragraph, rendering **bold** spans as bold runs."""
    pos = 0
    for m in _BOLD.finditer(text):
        if m.start() > pos:
            par.add_run(text[pos:m.start()])
        par.add_run(m.group(1)).bold = True
        pos = m.end()
    if pos < len(text):
        par.add_run(text[pos:])


def main():
    doc = Document()
    doc.styles["Normal"].font.size = Pt(11)
    with open(MD, encoding="utf-8") as f:
        lines = f.read().splitlines()

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        fig = _FIG.match(line.strip())
        if fig:
            path = os.path.join(FIGS, fig.group(1))
            if os.path.exists(path):
                doc.add_picture(path, width=Inches(6.0))
            else:
                doc.add_paragraph(f"[missing figure: {fig.group(1)}]")
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif line.startswith("> "):
            p = doc.add_paragraph(style="Intense Quote")
            add_runs(p, line[2:])
        else:
            add_runs(doc.add_paragraph(), line)

    doc.save(DOCX)
    n_imgs = sum(1 for _ in doc.inline_shapes)
    print(f"wrote {DOCX}  ({len(doc.paragraphs)} paragraphs, {n_imgs} figures)")


if __name__ == "__main__":
    main()
