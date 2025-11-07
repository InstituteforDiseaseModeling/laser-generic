# docs/gen-files.py
# from __future__ import annotations
# import importlib, pkgutil
# from pathlib import Path
# import mkdocs_gen_files as gen

#LOCAL_GENERIC_ROOT = Path("src/laser/generic")   # local code
# INSTALLED_CORE_PKG = "laser.core"                # installed package

# emitted: set[str] = set()

# def emit_page(dotted: str) -> None:
#     doc_path = Path("reference", *dotted.split("."), "index.md")
#     with gen.open(doc_path, "w") as f:
#         f.write(f"# {dotted}\n\n")
#         f.write(f"::: {dotted}\n")
#         f.write("    options:\n")
#         f.write("      show_root_heading: true\n")
#         f.write("      members_order: alphabetical\n")
#         f.write("      show_source: false\n")
#     emitted.add(dotted)

# ---------- laser.core (installed) ----------
# emit_page(INSTALLED_CORE_PKG)
# _core = importlib.import_module(INSTALLED_CORE_PKG)
# for _, name, _ in pkgutil.walk_packages(_core.__path__, prefix=INSTALLED_CORE_PKG + "."):
#     emit_page(name)

# ---------- laser.generic (local) ----------
#emit_page("laser.generic")

# Package pages for any directory that has a .py *anywhere* under it
#for d in sorted(p for p in LOCAL_GENERIC_ROOT.rglob("*") if p.is_dir()):
#    try:
#        has_py_desc = any(d.rglob("*.py"))
#    except (PermissionError, FileNotFoundError):
#        has_py_desc = False
#    if not has_py_desc:
#        continue
#    rel = d.relative_to(LOCAL_GENERIC_ROOT)
#    if str(rel) == ".":
#        continue
#    dotted = "laser.generic." + rel.as_posix().replace("/", ".")
#    emit_page(dotted)

# Module pages for every .py (excluding __init__.py)
#for py in sorted(LOCAL_GENERIC_ROOT.rglob("*.py")):
##    if py.name == "__init__.py":
 #       continue
#    rel = py.relative_to(LOCAL_GENERIC_ROOT).with_suffix("")
#    dotted = "laser.generic." + rel.as_posix().replace("/", ".")
#    emit_page(dotted)

# ---------- Build the sidebar (SUMMARY.md) ----------
# def depth(dotted: str) -> int:
#     return dotted.count(".")

# Ensure parents appear before children
# items = sorted(emitted, key=lambda s: (depth(s), s))

# ---------- Build the sidebar (grouped by top-level package) ----------
# --- after you’ve filled an `emitted` set of dotted names ---

# def write_group(prefix: str, all_items: set[str]) -> list[str]:
#     lines = []
#     root_path = prefix.replace(".", "/")
#     lines.append(f"- [{prefix}]({root_path})\n")
#     base = prefix.count(".")
#     for name in sorted([n for n in all_items if n.startswith(prefix + ".")], key=lambda s: (s.count("."), s)):
#         indent = "  " * (name.count(".") - base)
#         path = name.replace(".", "/")
#         lines.append(f"{indent}- [{name}]({path})\n")
#     return lines

# core_items = {n for n in emitted if n == "laser.core" or n.startswith("laser.core.")}
#generic_items = {n for n in emitted if n == "laser.generic" or n.startswith("laser.generic.")}

# with gen.open("reference/SUMMARY.md", "w") as nav:
#     nav.write("# API reference\n")
#     nav.writelines(write_group("laser.core", core_items))
#    nav.writelines(write_group("laser.generic", generic_items))


# TBD figure out how to get the navigation to look nice again.

# docs/gen-files.py
# import importlib, pkgutil, pathlib
# import mkdocs_gen_files as gen

# PKG = "laser.core"  # installed package

# def emit(mod: str):
#     doc_path = pathlib.Path("reference", *mod.split("."), "index.md")
#     with gen.open(doc_path, "w") as f:
#         f.write(f"# {mod}\n\n")
#         f.write(f"::: {mod}\n")
#         f.write("    options:\n")
#         f.write("      show_root_heading: true\n")
#         f.write("      members_order: alphabetical\n")
#         f.write("      show_source: false\n")

# # include the package root page
# emit(PKG)

# # walk all submodules found on the package __path__
# pkg = importlib.import_module(PKG)
# for _, name, _ in pkgutil.walk_packages(pkg.__path__, prefix=PKG + "."):
#     emit(name)



# docs/gen-files.py
# from __future__ import annotations
# import importlib
# import pkgutil
# from pathlib import PurePosixPath
# import mkdocs_gen_files as gen

# # The installed / importable package we want to document
# PACKAGE_NAME = "laser.core"

# # Keep track of everything we emit
# emitted: set[str] = set()

# def _doc_path_for(dotted: str) -> str:
#     """Return a normalized POSIX path for mkdocs-gen-files."""
#     return str(PurePosixPath("reference") / "/".join(dotted.split(".")) / "index.md")

# def emit_page(dotted: str) -> None:
#     """Write a virtual Markdown file for the given dotted module path."""
#     with gen.open(_doc_path_for(dotted), "w") as f:
#         f.write(f"# {dotted}\n\n")
#         f.write(f"::: {dotted}\n")
#         f.write("    options:\n")
#         f.write("      show_root_heading: true\n")
#         f.write("      members_order: alphabetical\n")
#         f.write("      show_source: false\n")
#     emitted.add(dotted)

# # ------------------ laser.core ------------------
# # Emit root page and every submodule discovered by pkgutil
# emit_page(PACKAGE_NAME)
# pkg = importlib.import_module(PACKAGE_NAME)
# for _, name, _ in pkgutil.walk_packages(pkg.__path__, prefix=PACKAGE_NAME + "."):
#     emit_page(name)

# # ------------------ Build SUMMARY.md ------------------
# def write_summary(prefix: str, items: set[str]) -> list[str]:
#     """Create a markdown nav tree for all emitted modules under prefix."""
#     lines: list[str] = []
#     base_depth = prefix.count(".")
#     for name in sorted(items, key=lambda s: (s.count("."), s)):
#         depth = name.count(".") - base_depth
#         indent = "  " * depth
#         path = name.replace(".", "/") + "/index.md"
#         lines.append(f"{indent}- [{name}]({path})\n")
#     return lines

# with gen.open("reference/SUMMARY.md", "w") as nav:
#     nav.write("# API reference\n")
#     nav.writelines(write_summary(PACKAGE_NAME, emitted))

# docs/gen-files.py
# from __future__ import annotations
# import importlib, pkgutil
# from pathlib import PurePosixPath
# import mkdocs_gen_files as gen

# PACKAGE_NAME = "laser.core"

# emitted: set[str] = set()

# def _doc_path_for(dotted: str) -> str:
#     return str(PurePosixPath("reference") / "/".join(dotted.split(".")) / "index.md")

# def emit_page(dotted: str, body: str | None = None) -> None:
#     with gen.open(_doc_path_for(dotted), "w") as f:
#         if body is None:
#             f.write(f"# {dotted}\n\n")
#             f.write(f"::: {dotted}\n")
#             f.write("    options:\n")
#             f.write("      show_root_heading: true\n")
#             f.write("      members_order: alphabetical\n")
#             f.write("      show_source: false\n")
#         else:
#             f.write(body)
#     emitted.add(dotted)

# # --- create a real parent node so it can appear in the TOC as 'laser'
# emit_page("laser", "# laser\n\nThis is the root namespace for LASER packages.\n")

# # --- laser.core (installed)
# emit_page(PACKAGE_NAME)
# pkg = importlib.import_module(PACKAGE_NAME)
# for _, name, _ in pkgutil.walk_packages(pkg.__path__, prefix=PACKAGE_NAME + "."):
#     emit_page(name)

# # --- Build grouped SUMMARY.md that nests under 'laser'
# def write_group(label: str, children: list[tuple[str, list[str]]]) -> list[str]:
#     """
#     label: top-level label ('laser')
#     children: list of (root_module, descendants) where descendants are dotted names
#     """
#     lines: list[str] = []
#     # parent 'laser' (relative to reference/)
#     lines.append(f"- [{label}](laser/index.md)\n")
#     for root, desc in children:
#         # root entry (e.g. laser.core)
#         lines.append(f"  - [{root}]({root.replace('.', '/')}/index.md)\n")
#         base = root.count(".")
#         for name in sorted(desc, key=lambda s: (s.count("."), s)):
#             depth = name.count(".") - base + 2  # +2 because we're under 'laser' then root
#             indent = "  " * depth
#             path = name.replace(".", "/") + "/index.md"
#             lines.append(f"{indent}- [{name}]({path})\n")
#     return lines
# emit_page("laser.generic")

# core_items = sorted(n for n in emitted if n.startswith("laser.core.") )
# # We don't generate generic pages here—autonav will—but we want the root link visible.
# generic_items: list[str] = []  # keep empty; the root page will exist via autonav or your own page

# with gen.open("reference/SUMMARY.md", "w") as nav:
#     nav.write("# API reference\n")
#     nav.writelines(write_group("laser", [
#         ("laser.core", core_items),
#         ("laser.generic", generic_items),   # just the root; its subtree will render within page
#     ]))

# docs/gen-files.py
# from __future__ import annotations
# import importlib
# import inspect
# import pkgutil
# from pathlib import PurePosixPath
# import mkdocs_gen_files as gen

# PACKAGE = "laser.core"
# INCLUDE_PRIVATE = True  # set False to hide _private names

# def _doc_path_for(dotted: str) -> str:
#     return str(PurePosixPath("reference") / "/".join(dotted.split(".")) / "index.md")

# def _emit_page(dotted: str, body: str | None = None) -> None:
#     with gen.open(_doc_path_for(dotted), "w") as f:
#         if body is None:
#             f.write(f"# {dotted}\n\n")
#             f.write(f"::: {dotted}\n")
#             f.write("    options:\n")
#             f.write("      show_root_heading: true\n")
#             f.write("      members_order: alphabetical\n")
#             f.write("      show_source: false\n")
#         else:
#             f.write(body)

# def _is_dunder(n: str) -> bool:
#     return len(n) > 4 and n.startswith("__") and n.endswith("__")

# def _keep(n: str) -> bool:
#     if _is_dunder(n):
#         return False
#     return INCLUDE_PRIVATE or not n.startswith("_")

# emitted: set[str] = set()
# def emit(dotted: str, body: str | None = None) -> None:
#     if dotted in emitted:
#         return
#     _emit_page(dotted, body)
#     emitted.add(dotted)

# # parent 'laser' so we can nest under it in SUMMARY
# emit("laser", "# laser\n\nRoot namespace for LASER packages.\n")

# # walk modules
# emit(PACKAGE)
# pkg = importlib.import_module(PACKAGE)

# for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix=PACKAGE + "."):
#     emit(modname)  # module page

#     try:
#         m = importlib.import_module(modname)
#     except Exception:
#         continue  # keep the module page; skip per-object pages if import fails

#     # classes
#     for name, obj in inspect.getmembers(m, inspect.isclass):
#         if not _keep(name):
#             continue
#         if getattr(obj, "__module__", None) != modname:
#             continue
#         emit(
#             f"{modname}.{name}",
#             body=(
#                 f"# {modname}.{name}\n\n"
#                 f"::: {modname}.{name}\n"
#                 "    options:\n"
#                 "      show_root_heading: true\n"
#                 "      members_order: alphabetical\n"
#                 "      show_source: false\n"
#             ),
#         )

#     # functions
#     for name, obj in inspect.getmembers(m, inspect.isfunction):
#         if not _keep(name):
#             continue
#         if getattr(obj, "__module__", None) != modname:
#             continue
#         emit(
#             f"{modname}.{name}",
#             body=(
#                 f"# {modname}.{name}\n\n"
#                 f"::: {modname}.{name}\n"
#                 "    options:\n"
#                 "      show_root_heading: true\n"
#                 "      members_order: alphabetical\n"
#                 "      show_source: false\n"
#             ),
#         )

# # SUMMARY.md (under reference/laser)
# def write_summary_under_laser(items: list[str]) -> list[str]:
#     lines: list[str] = []
#     lines.append(f"- [laser](laser/index.md)\n")
#     lines.append(f"  - [laser.core](laser/core/index.md)\n")
#     base = PACKAGE.count(".")
#     for full in sorted(items, key=lambda s: (s.count("."), s)):
#         if full == PACKAGE:
#             continue
#         depth = full.count(".") - base + 2
#         indent = "  " * depth
#         path = full.replace(".", "/") + "/index.md"
#         lines.append(f"{indent}- [{full}]({path})\n")
#     return lines

# core_items = [n for n in emitted if n == PACKAGE or n.startswith(PACKAGE + ".")]
# with gen.open("reference/SUMMARY.md", "w") as nav:
#     nav.write("# API reference\n")
#     nav.writelines(write_summary_under_laser(core_items))

# docs/_scripts/generate_api_pages.py
# from __future__ import annotations
# import os
# from pathlib import Path
# import mkdocs_gen_files as gen

# SRC = Path("src")
# NAMESPACE = "laser"  # <-- your shared namespace root

# ns_root = SRC / NAMESPACE

# # Collect all importable module/package paths under the namespace
# # We treat any directory with Python files (or __init__.py) as a module page,
# # and also include top-level packages as index pages.
# modules: list[str] = []

# for path in ns_root.rglob("*"):
#     if path.is_dir():
#         # include a package if it has __init__.py or any .py children
#         if (path / "__init__.py").exists() or any(p.suffix == ".py" for p in path.iterdir()):
#             pkg = path.relative_to(SRC).as_posix().replace("/", ".")
#             modules.append(pkg)
#     elif path.suffix == ".py" and path.name != "__init__.py":
#         mod = path.relative_to(SRC).with_suffix("").as_posix().replace("/", ".")
#         modules.append(mod)

# # Ensure deterministic order and that parents appear before children
# modules = sorted(set(modules), key=lambda s: (s.count("."), s))

# # Write a clean “Reference” landing page
# with gen.open("reference/index.md", "w") as f:
#     f.write(f"# {NAMESPACE} reference\n\n")
#     f.write("This section is auto-generated from the source under `src/`.\n\n")

# # Build a nested SUMMARY.md for nice sidebar sections
# summary_lines = ["# Reference\n", f"- [{NAMESPACE}](index.md)"]

# # Emit one page per module/package
# for dotted in modules:
#     # file path under docs/reference
#     doc_path = Path("reference", *dotted.split("."), "index.md")
#     with gen.open(doc_path, "w") as f:
#         title = dotted.split(".")[-1]
#         f.write(f"# {dotted}\n\n")
#         f.write("::: " + dotted + "\n")
#         f.write("    options:\n")
#         f.write("      show_root_heading: true\n")
#         f.write("      members_order: alphabetical\n")
#         f.write("      show_source: false\n")

#     # add to SUMMARY with nested folders
#     parts = dotted.split(".")
#     indent = "  " * (len(parts) - 1)
#     summary_lines.append(f"{indent}- [{parts[-1]}]({dotted.replace('.', '/')}/index.md)")

# # Write the summary nav file consumed by mkdocs-literate-nav
# with gen.open("reference/SUMMARY.md", "w") as f:
#     f.write("\n".join(summary_lines) + "\n")

# docs/gen-files.py
# from __future__ import annotations
# import importlib
# import pkgutil
# from pathlib import Path, PurePosixPath
# import mkdocs_gen_files as gen

# SRC = Path("src")
# NAMESPACE = "laser"              # shared namespace root on disk
# NS_ROOT = SRC / NAMESPACE

# CORE_PKG = "laser.core"          # installed package to include

# # --- helpers ----------------------------------------------------
# def posix_ref_for(dotted: str) -> str:
#     # reference/<dotted>/index.md using forward slashes (Windows-safe)
#     return str(PurePosixPath("reference") / "/".join(dotted.split(".")) / "index.md")

# def emit_page(dotted: str, body: str | None = None) -> None:
#     with gen.open(posix_ref_for(dotted), "w") as f:
#         if body is None:
#             f.write(f"# {dotted}\n\n")
#             f.write(f"::: {dotted}\n")
#             f.write("    options:\n")
#             f.write("      show_root_heading: true\n")
#             f.write("      members_order: alphabetical\n")
#             f.write("      show_source: false\n")
#         else:
#             f.write(body)

# # --- 1) Collect modules from src/laser/** (laser generic & any others on disk) ---
# disk_modules: set[str] = set()

# # Include directories that are packages OR have any .py anywhere under them (implicit namespaces)
# for d in NS_ROOT.rglob("*"):
#     if d.is_dir():
#         try:
#             has_py = any(d.rglob("*.py"))
#         except (PermissionError, FileNotFoundError):
#             has_py = False
#         if has_py:
#             dotted = d.relative_to(SRC).as_posix().replace("/", ".")
#             disk_modules.add(dotted)
#     elif d.suffix == ".py" and d.name != "__init__.py":
#         mod = d.relative_to(SRC).with_suffix("").as_posix().replace("/", ".")
#         disk_modules.add(mod)

# # Ensure parents before children
# disk_modules = set(sorted(disk_modules, key=lambda s: (s.count("."), s)))

# # --- 2) Collect modules from installed laser.core -------------------------------
# core_modules: set[str] = set()
# try:
#     core_pkg = importlib.import_module(CORE_PKG)
#     core_modules.add(CORE_PKG)
#     for _, name, _ in pkgutil.walk_packages(core_pkg.__path__, prefix=CORE_PKG + "."):
#         core_modules.add(name)
# except Exception:
#     # If laser.core isn't importable, keep the rest; SUMMARY will just omit it
#     core_modules = set()

# # --- 3) Emit pages for everything we discovered --------------------------------
# emitted: set[str] = set()

# def emit_once(dotted: str, body: str | None = None) -> None:
#     if dotted in emitted:
#         return
#     emit_page(dotted, body)
#     emitted.add(dotted)

# # Optional: a Reference landing page (not required, but nice to have)
# emit_once(f"{NAMESPACE}", f"# {NAMESPACE} reference\n\nThis section is auto-generated.\n\n")

# for dotted in sorted(disk_modules, key=lambda s: (s.count("."), s)):
#     emit_once(dotted)

# for dotted in sorted(core_modules, key=lambda s: (s.count("."), s)):
#     emit_once(dotted)

# # --- 4) Write SUMMARY.md so TOC shows *siblings* 'laser.core' and 'laser.generic' ----
# def write_group(root: str, pool: set[str]) -> list[str]:
#     """
#     Make a block with a top-level entry for `root` (e.g. 'laser.core')
#     and all its descendants, with paths relative to 'reference/'.
#     """
#     lines: list[str] = []
#     if root in pool or any(x.startswith(root + ".") for x in pool):
#         # top-level sibling under API reference
#         lines.append(f"- [{root}]({root.replace('.', '/')}/index.md)\n")
#         base = root.count(".")
#         descendants = sorted(
#             (n for n in pool if n != root and n.startswith(root + ".")),
#             key=lambda s: (s.count("."), s),
#         )
#         for name in descendants:
#             indent = "  " * (name.count(".") - base)
#             path = name.replace(".", "/") + "/index.md"
#             lines.append(f"{indent}- [{name}]({path})\n")
#     return lines

# # Build the two sibling groups explicitly
# summary_lines: list[str] = ["# API reference\n"]

# # laser.core block (from installed package)
# summary_lines += write_group("laser.core", emitted)

# # laser.generic block (from src tree)
# summary_lines += write_group("laser.generic", emitted)

# with gen.open("reference/SUMMARY.md", "w") as f:
#     f.write("".join(summary_lines))
# docs/gen-files.py
# from __future__ import annotations
# import importlib
# import pkgutil
# from pathlib import Path, PurePosixPath
# import mkdocs_gen_files as gen

# SRC = Path("src")
# NS = "laser"
# NS_ROOT = SRC / NS
# CORE_ROOT = "laser.core"  # installed package

# # ---------- helpers ----------
# def ref_md(dotted: str) -> str:
#     """reference/<dotted>/index.md with POSIX separators."""
#     return str(PurePosixPath("reference") / "/".join(dotted.split(".")) / "index.md")

# def emit_page(dotted: str, body: str | None = None) -> None:
#     with gen.open(ref_md(dotted), "w") as f:
#         if body is None:
#             f.write(f"# {dotted}\n\n")
#             f.write(f"::: {dotted}\n")
#             f.write("    options:\n")
#             f.write("      show_root_heading: true\n")
#             f.write("      members_order: alphabetical\n")
#             f.write("      show_source: false\n")
#         else:
#             f.write(body)

# # ---------- collect: disk (laser.generic and any other src/laser/*) ----------
# packages: set[str] = set()
# modules: set[str] = set()

# # 1) one module page per .py (excluding __init__.py)
# for py in NS_ROOT.rglob("*.py"):
#     if py.name == "__init__.py":
#         continue
#     rel = py.relative_to(SRC).with_suffix("")              # e.g. laser/generic/models/SEIR
#     dotted = rel.as_posix().replace("/", ".")              # -> laser.generic.models.SEIR
#     modules.add(dotted)

#     # 2) add all ancestor packages up to 'laser'
#     parent = py.parent
#     while parent != SRC and parent != SRC.parent:
#         rel_pkg = parent.relative_to(SRC).as_posix().replace("/", ".")
#         packages.add(rel_pkg)                               # e.g. laser.generic.models
#         if rel_pkg == NS:                                   # stop at 'laser'
#             break
#         parent = parent.parent

# # Ensure the root 'laser' page exists (nice landing + required for TOC nesting)
# packages.add(NS)

# # ---------- collect: installed laser.core ----------
# core_items: set[str] = set()
# try:
#     core_pkg = importlib.import_module(CORE_ROOT)
#     core_items.add(CORE_ROOT)
#     for _, name, _ in pkgutil.walk_packages(core_pkg.__path__, prefix=CORE_ROOT + "."):
#         core_items.add(name)
# except Exception:
#     # If not importable, we just won't emit it
#     pass

# # ---------- emit all pages ----------
# emitted: set[str] = set()

# def emit_once(dotted: str, body: str | None = None) -> None:
#     if dotted in emitted:
#         return
#     emit_page(dotted, body)
#     emitted.add(dotted)

# # Optional: small landing page for 'laser'
# emit_once(NS, f"# {NS}\n\nRoot namespace for LASER packages.\n")

# for name in sorted(packages, key=lambda s: (s.count("."), s)):
#     emit_once(name)

# for name in sorted(modules, key=lambda s: (s.count("."), s)):
#     emit_once(name)

# for name in sorted(core_items, key=lambda s: (s.count("."), s)):
#     emit_once(name)

# # ---------- SUMMARY.md (two siblings: laser.core & laser.generic) ----------
# def write_block(root: str, pool: set[str]) -> list[str]:
#     """root ('laser.core' or 'laser.generic'), plus all its descendants."""
#     lines: list[str] = []
#     # write the root only if present
#     if root in pool or any(x.startswith(root + ".") for x in pool):
#         lines.append(f"- [{root}]({root.replace('.', '/')}/index.md)\n")
#         base = root.count(".")
#         children = sorted(
#             (n for n in pool if n != root and n.startswith(root + ".")),
#             key=lambda s: (s.count("."), s),
#         )
#         for n in children:
#             indent = "  " * (n.count(".") - base)
#             path = n.replace(".", "/") + "/index.md"
#             lines.append(f"{indent}- [{n}]({path})\n")
#     return lines

# summary: list[str] = ["# API reference\n"]
# # Use 'emitted' as the pool so both disk + core show up
# summary += write_block("laser.core", emitted)
# summary += write_block("laser.generic", emitted)

# with gen.open("reference/SUMMARY.md", "w") as f:
#     f.write("".join(summary))

# docs/gen-files.py
from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path, PurePosixPath
import mkdocs_gen_files as gen

SRC = Path("src")
NS = "laser"
NS_ROOT = SRC / NS
CORE_ROOT = "laser.core"  # importable package

def ref_md(dotted: str) -> str:
    """reference/<dotted>/index.md with POSIX separators."""
    return str(PurePosixPath("reference") / "/".join(dotted.split(".")) / "index.md")

def emit_page(dotted: str, body: str | None = None) -> None:
    with gen.open(ref_md(dotted), "w") as f:
        if body is None:
            f.write(f"# {dotted}\n\n")
            f.write(f"::: {dotted}\n")
            f.write("    options:\n")
            f.write("      show_root_heading: true\n")
            f.write("      members_order: alphabetical\n")
            f.write("      show_source: false\n")
        else:
            f.write(body)

# ---------- collect from disk: every module + all ancestor packages ----------
packages: set[str] = set()
modules: set[str]  = set()

# Ensure the root 'laser' landing exists so we can nest under it if desired
packages.add(NS)

for py in NS_ROOT.rglob("*.py"):
    if py.name == "__init__.py":
        # still create the package page for its directory
        pkg = py.parent.relative_to(SRC).as_posix().replace("/", ".")
        packages.add(pkg)
        continue

    # module page for each file
    rel_mod = py.relative_to(SRC).with_suffix("")
    dotted_mod = rel_mod.as_posix().replace("/", ".")
    modules.add(dotted_mod)

    # ancestor packages up to 'laser'
    p = py.parent
    while True:
        rel_pkg = p.relative_to(SRC).as_posix().replace("/", ".")
        packages.add(rel_pkg)
        if rel_pkg == NS:
            break
        p = p.parent

# ---------- collect from installed laser.core ----------
core_items: set[str] = set()
try:
    core_pkg = importlib.import_module(CORE_ROOT)
    core_items.add(CORE_ROOT)
    for _, name, _ in pkgutil.walk_packages(core_pkg.__path__, prefix=CORE_ROOT + "."):
        core_items.add(name)
except Exception:
    # not importable in this env; fine
    pass

# ---------- emit all pages ----------
emitted: set[str] = set()
def emit_once(dotted: str, body: str | None = None) -> None:
    if dotted in emitted:
        return
    emit_page(dotted, body)
    emitted.add(dotted)

# small landing for 'laser'
emit_once(NS, f"# {NS}\n\nRoot namespace for LASER packages.\n")

for name in sorted(packages, key=lambda s: (s.count("."), s)):
    emit_once(name)

for name in sorted(modules, key=lambda s: (s.count("."), s)):
    emit_once(name)

for name in sorted(core_items, key=lambda s: (s.count("."), s)):
    emit_once(name)

# ---------- SUMMARY.md (two siblings, with correct hierarchical ordering) ----------

def _build_tree(root: str, names: set[str]) -> dict:
    """
    Build a nested dict representing the subtree under `root`.
    Ensures all intermediate parents are present.
    Example tree keys are the *segment* names (e.g., 'generic', 'models', 'SEIR').
    """
    root_parts = root.split(".")
    tree: dict = {}

    def insert(parts: list[str]) -> None:
        node = tree
        for seg in parts:
            node = node.setdefault(seg, {})

    # Consider only entries under the root (including root itself, though we'll render root separately)
    for full in (n for n in names if n == root or n.startswith(root + ".")):
        parts = full.split(".")
        # Add every prefix between root and the leaf
        for i in range(len(root_parts) + 1, len(parts) + 1):
            insert(parts[len(root_parts):i])  # relative parts under root

    return tree

def _render_tree(root: str, tree: dict) -> list[str]:
    """
    Render the nested dict as literate-nav bullets under `root`.
    Outputs links relative to 'reference/' using '.../index.md'.
    """
    lines: list[str] = []

    def walk(rel_parts: list[str], node: dict):
        # rel_parts is the path under the root (e.g., ['generic', 'models'])
        dotted = ".".join([root] + rel_parts)
        depth = len(rel_parts)  # depth under the root
        indent = "  " * (depth)  # bullet indent under the root line we'll add earlier
        path = "/".join([root.replace(".", "/")] + rel_parts) + "/index.md"
        lines.append(f"{indent}- [{dotted}]({path})\n")
        for seg in sorted(node.keys()):
            walk(rel_parts + [seg], node[seg])

    # walk each top-level child under the root
    for seg in sorted(tree.keys()):
        walk([seg], tree[seg])

    return lines

def _write_block(root: str, pool: set[str]) -> list[str]:
    """
    Write the top-level root entry and a properly ordered subtree beneath it.
    """
    # Top-level root entry (sibling under API reference)
    out: list[str] = [f"- [{root}]({root.replace('.', '/')}/index.md)\n"]
    subtree = _build_tree(root, pool)
    out.extend(_render_tree(root, subtree))
    return out

# Build SUMMARY with two sibling blocks
summary_lines: list[str] = ["# API reference\n"]
summary_lines += _write_block("laser.core", emitted)     # emitted contains both disk + core
summary_lines += _write_block("laser.generic", emitted)

with gen.open("reference/SUMMARY.md", "w") as f:
    f.write("".join(summary_lines))
