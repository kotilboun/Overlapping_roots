#!/usr/bin/env python3
"""Produce 600-dpi LZW-compressed TIFF submission copies of every figure.

Mathematical Biosciences (Springer) follows the standard Springer
Nature artwork guidelines: combination art (color figures mixing line work,
text, and shading -- which all nine figures here are) requires a minimum
resolution of 600 dpi, and TIFF is the preferred raster format.

pdfLaTeX cannot embed .tiff files directly (\\includegraphics raises
"Unknown graphics extension: .tiff"), so the manuscript continues to
\\includegraphics the native Fig*.pdf / Fig5.png files that each
Figure_N/02_make_figure.py already writes -- those compile manuscript.pdf.
This script separately produces Fig*.tiff at 600 dpi, RGB, LZW-compressed,
as the artwork files to upload to the journal submission system alongside
the LaTeX source; for Figures 1, 3, 4, 6, and 7 it is a faithful
high-resolution rasterization of exactly what appears in the compiled
manuscript (rendered from the same PDF), not a re-plot. Figure 5 is
converted directly from its already-600-dpi PNG. Figures 2, 8, and 9 already
have their own native 600-dpi TIFF (Figure 2 copied from
`supplementary_figures/suppl_FigS6/`; Figures 8 and 9 written by their own
02_make_figure.py) and are left untouched here.

Run after all Figure_N/02_make_figure.py have populated ../figures/:
    python make_bmb_tiffs.py
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE.parent / "figures"
TARGET_DPI = 600
PDF_FIGURES = (1, 3, 4, 6, 7)
PNG_FIGURES = (5,)


def pdf_to_tiff(number: int) -> Path:
    pdf_path = FIGURES_DIR / f"Fig{number}.pdf"
    scale = TARGET_DPI / 72.0
    with fitz.open(pdf_path) as document:
        page = document[0]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    tiff_path = FIGURES_DIR / f"Fig{number}.tiff"
    image.save(tiff_path, compression="tiff_lzw", dpi=(TARGET_DPI, TARGET_DPI))
    return tiff_path


def png_to_tiff(number: int) -> Path:
    png_path = FIGURES_DIR / f"Fig{number}.png"
    tiff_path = FIGURES_DIR / f"Fig{number}.tiff"
    with Image.open(png_path) as image:
        image.convert("RGB").save(tiff_path, compression="tiff_lzw", dpi=(TARGET_DPI, TARGET_DPI))
    return tiff_path


def main() -> None:
    written = []
    for number in PDF_FIGURES:
        written.append(pdf_to_tiff(number))
    for number in PNG_FIGURES:
        written.append(png_to_tiff(number))
    for path in sorted(written, key=lambda p: int("".join(filter(str.isdigit, p.stem)))):
        with Image.open(path) as image:
            width, height = image.size
        size_kb = path.stat().st_size / 1024
        print(f"{path.name}: {width}x{height} px @ {TARGET_DPI} dpi, {size_kb:,.0f} KB")


if __name__ == "__main__":
    main()
