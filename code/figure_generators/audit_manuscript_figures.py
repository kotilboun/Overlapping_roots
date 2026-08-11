"""Audit generated figure integration in the compiled manuscript PDF.

The script is intentionally independent of the numerical generators.  It checks
that each numbered figure caption is present exactly once and can optionally
render the nine caption pages for a quick visual inspection.

Matching note: a caption block reads "Fig. N <Capitalized title>" (e.g.
"Fig. 9 Representative..."), whereas an inline cross-reference such as
"...reported in Fig.~\\ref{fig:fig7}" renders as "Fig. 7" followed by a
lowercase word or sentence-ending punctuation (e.g. "Fig. 7 and", "Fig. 7.").
Since Figures 8 and 9 cross-reference Figure 7 (and each other) in their own
captions, a bare substring search over-counts; requiring an uppercase letter
immediately after "Fig. N " distinguishes the actual caption from these
cross-references.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "manuscript.pdf",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        help="Render each figure-caption page as a PNG in this directory.",
    )
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    document = fitz.open(pdf_path)
    captions: dict[str, list[int]] = {}

    for figure_number in range(1, 10):
        pattern = re.compile(rf"Fig\. {figure_number} [A-Z]")
        pages = [
            page_number + 1
            for page_number, page in enumerate(document)
            if pattern.search(page.get_text())
        ]
        captions[str(figure_number)] = pages

    report = {
        "pdf": str(pdf_path),
        "page_count": document.page_count,
        "captions": captions,
        "all_captions_unique": all(len(pages) == 1 for pages in captions.values()),
    }

    if args.render_dir:
        render_dir = args.render_dir.resolve()
        render_dir.mkdir(parents=True, exist_ok=True)
        rendered: dict[str, str] = {}
        for figure_number, pages in captions.items():
            if len(pages) != 1:
                continue
            page_number = pages[0]
            output = render_dir / f"figure_{figure_number}_manuscript_page_{page_number}.png"
            page = document[page_number - 1]
            page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(output)
            rendered[figure_number] = str(output)
        report["rendered_pages"] = rendered

    print(json.dumps(report, indent=2))
    return 0 if report["all_captions_unique"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
