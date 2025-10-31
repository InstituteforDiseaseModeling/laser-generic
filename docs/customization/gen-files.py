# docs/gen-files.py
import importlib, pkgutil, pathlib
import mkdocs_gen_files as gen

PKG = "laser.core"  # installed package

def emit(mod: str):
    doc_path = pathlib.Path("reference", *mod.split("."), "index.md")
    with gen.open(doc_path, "w") as f:
        f.write(f"# {mod}\n\n")
        f.write(f"::: {mod}\n")
        f.write("    options:\n")
        f.write("      show_root_heading: true\n")
        f.write("      members_order: alphabetical\n")
        f.write("      show_source: false\n")

# include the package root page
emit(PKG)

# walk all submodules found on the package __path__
pkg = importlib.import_module(PKG)
for _, name, _ in pkgutil.walk_packages(pkg.__path__, prefix=PKG + "."):
    emit(name)
