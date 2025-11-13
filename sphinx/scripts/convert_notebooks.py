# sphinx/scripts/convert_notebooks.py

import pathlib
import subprocess

here = pathlib.Path(__file__).resolve().parent
root = here.parent.parent  # → /docs

nb_root = root / "docs" / "notebooks"
rst_root = here.parent / "source" / "converted" / "notebooks"

rst_root.mkdir(parents=True, exist_ok=True)

for nb in nb_root.glob("*.ipynb"):
    out = rst_root / nb.with_suffix(".rst").name
    print(f"Converting {nb} → {out}")
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "rst",
            # "--execute",  # Uncomment to run notebooks before converting
            "--output",
            out.name,
            "--output-dir",
            str(rst_root),
            str(nb),
        ],
        check=True,
    )
