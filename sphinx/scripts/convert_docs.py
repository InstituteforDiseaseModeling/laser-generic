# scripts/convert_docs.py
import pathlib
import subprocess

# Paths relative to the current script file (not current working directory)
here = pathlib.Path(__file__).resolve().parent
root = here.parent.parent  # → /docs

md_dir = root / "docs"  # Was: pathlib.Path("docs")
rst_dir = here.parent / "source" / "converted"  # → /docs/sphinx/source/converted

rst_dir.mkdir(parents=True, exist_ok=True)

for md in md_dir.rglob("*.md"):
    rel = md.relative_to(md_dir)
    out = rst_dir / rel.with_suffix(".rst")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "pandoc", "-f", "markdown", "-t", "rst",
        str(md), "-o", str(out)
    ])
