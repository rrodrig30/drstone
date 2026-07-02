"""Render a docs/*.md manuscript to a print-ready PDF with embedded figures.

Pipeline: markdown -> styled HTML (python-markdown; tables + [[FIGURE:]] handled)
-> Chromium --headless --print-to-pdf. No pandoc/LaTeX needed.

Run:  python docs/build_pdf.py <stem>       e.g. 01_DrStone_Methods
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = "/home/exx/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome"

CSS = """
@page { size: Letter; margin: 0.9in; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt;
       line-height: 1.5; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 6pt; line-height: 1.25; }
h2 { font-size: 13pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt;
     margin: 16pt 0 6pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
p { margin: 5pt 0; text-align: justify; }
img { display: block; max-width: 92%; margin: 10pt auto 2pt; }
img, table, figure { page-break-inside: avoid; }
/* caption = the bold paragraph immediately after a figure image */
img + p, p:has(> strong:first-child) { }
table { border-collapse: collapse; width: 100%; font-size: 8pt; margin: 8pt 0; }
th, td { border: 1px solid #999; padding: 2.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #ededed; }
blockquote { margin: 8pt 0 8pt 14pt; padding-left: 10pt; border-left: 3px solid #ccc;
             color: #333; font-style: italic; }
"""


def build(stem: str) -> str:
    md_path = os.path.join(HERE, stem + ".md")
    pdf_path = os.path.join(HERE, stem + ".pdf")
    src = open(md_path, encoding="utf-8").read()

    def figrepl(m):
        p = os.path.join(HERE, "figures", m.group(1).strip())
        return f'\n\n<img src="file://{p}" />\n\n'

    src = re.sub(r"\[\[FIGURE:\s*(.+?)\s*\]\]", figrepl, src)
    body = markdown.markdown(src, extensions=["tables", "sane_lists", "attr_list"])
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    htmlf = os.path.join(tempfile.gettempdir(), stem + ".render.html")
    open(htmlf, "w", encoding="utf-8").write(html)
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
         "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
         "--virtual-time-budget=20000", f"--print-to-pdf={pdf_path}", f"file://{htmlf}"],
        check=True, capture_output=True)
    return pdf_path


def main():
    stems = sys.argv[1:] or ["01_DrStone_Methods"]
    for stem in stems:
        p = build(stem)
        print(f"wrote {p}  ({os.path.getsize(p)//1024} KB)")


if __name__ == "__main__":
    main()
