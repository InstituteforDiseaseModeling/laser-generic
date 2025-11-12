# scripts/convert_docs.py
import pathlib
import subprocess

md_dir = pathlib.Path("docs")
rst_dir = pathlib.Path("sphinx/source/converted")
rst_dir.mkdir(parents=True, exist_ok=True)

for md in md_dir.rglob("*.md"):
    rel = md.relative_to(md_dir)
    out = rst_dir / rel.with_suffix(".rst")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pandoc", "-f", "markdown", "-t", "rst", str(md), "-o", str(out)])
