"""Extract original Figure 3/4/5 with captions from the frozen official PDF.

Usage: python extract_deepseek_v41_report_figures.py report.pdf /path/to/pdftocairo
Requires pypdf and Poppler. Figures retain upstream text, colors and geometry.
Only surrounding body text and page margins are cropped; no raster editing.
"""
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from pypdf import PdfReader, PdfWriter

PDF_SHA256 = "ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d"
# Figure number, one-based PDF page, crop top/bottom in PDF points from page top.
FIGURES = [(3, 7, 82, 411), (4, 10, 80, 309), (5, 11, 80, 369)]

def main():
    source, cairo = sys.argv[1:]
    if sha256(Path(source).read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError("PDF differs from HF revision fb2764a5cf321eaa5070ca8f9e892818f477c16d")
    reader = PdfReader(source)
    dest = Path(__file__).resolve().parents[2] / "wiki/01_theory/01_models/deepseek/assets"
    dest.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as scratch:
        for number, page_number, top, bottom in FIGURES:
            page = reader.pages[page_number - 1]
            height = float(page.mediabox.height)
            page.mediabox.lower_left = (66, height - bottom)
            page.mediabox.upper_right = (529, height - top)
            page.cropbox = page.mediabox
            cropped = Path(scratch) / f"figure{number}.pdf"
            writer = PdfWriter()
            writer.add_page(page)
            writer.write(cropped)
            output = dest / f"deepseek_v41_report_figure{number}.svg"
            subprocess.run([cairo, "-svg", str(cropped), str(output)], check=True)
            print(output)

if __name__ == "__main__":
    main()
