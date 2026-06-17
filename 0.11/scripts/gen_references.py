"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

PACKAGE = "ecologits"

nav = mkdocs_gen_files.Nav()

for path in Path(PACKAGE).rglob("*.py"):
    module_path = path.relative_to(PACKAGE).with_suffix("")
    doc_path = path.relative_to(PACKAGE).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = list(module_path.parts)
    if parts[-1] == "__init__" or parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write("::: " + identifier)

    mkdocs_gen_files.set_edit_path(full_doc_path, Path("../") / path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
