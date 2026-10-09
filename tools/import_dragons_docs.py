#!/usr/bin/env python3
"""Refresh the DRAGONS reference without importing its scientific dependencies.

Run with the documentation environment::

    python tools/import_dragons_docs.py

The pinned revision is the source of the published DRAGONS Tools documentation.
An existing checkout can be supplied with --source-root for offline generation.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
import re
from textwrap import indent
from urllib.request import urlopen
import warnings

from sphinx.ext.napoleon.docstring import NumpyDocstring


REVISION = "b13161739bd20a47c87952cdc89f74af0f7a7e26"
REPOSITORY = "https://github.com/meraxes-devs/dragons"
RAW = f"https://raw.githubusercontent.com/meraxes-devs/dragons/{REVISION}"
PAGES = {
    "meraxes": [
        "dragons.meraxes.galaxy_history",
        "dragons.meraxes.io",
        "dragons.meraxes.reion",
        "dragons.meraxes.plots",
    ],
    "munge": ["dragons.munge.munge", "dragons.munge.regrid"],
    "nbody": ["dragons.nbody.io"],
    "plotutils": ["dragons.plotutils"],
}
# Members published on the four original API pages, in their displayed order.
MEMBERS = {
    "dragons.meraxes.galaxy_history": ["galaxy_history"],
    "dragons.meraxes.io": [
        "check_for_global_xH", "check_for_redshift", "grab_redshift",
        "grab_unsampled_snapshot", "list_grids", "read_descendant_indices",
        "read_firstprogenitor_indices", "read_gals", "read_git_info",
        "read_global_J_21", "read_global_xH", "read_grid", "read_input_params",
        "read_nextprogenitor_indices", "read_ps", "read_snaplist", "read_units",
        "set_little_h",
    ],
    "dragons.meraxes.reion": ["electron_optical_depth"],
    "dragons.meraxes.plots": [],
    "dragons.munge.munge": [
        "describe", "edges_to_centers", "mass_function", "ndarray_to_dataframe",
        "power_spectrum", "pretty_print_dict", "smooth_grid",
    ],
    "dragons.munge.regrid": ["regrid"],
    "dragons.nbody.io": ["read_density_grid", "read_grid", "read_halo_catalog"],
    "dragons.plotutils": ["density_contour"],
}


def source_path(module: str) -> str:
    suffix = ".pyx" if module == "dragons.munge.regrid" else ".py"
    return module.replace(".", "/") + suffix


def source_link(path: str, start: int | None = None, end: int | None = None) -> str:
    fragment = f"#L{start}-L{end}" if start is not None else ""
    return f"{REPOSITORY}/blob/{REVISION}/{path}{fragment}"


def heading(text: str, char: str = "=") -> str:
    return f"{text}\n{char * len(text)}\n\n"


def parse_module(source: str) -> ast.Module:
    # Some upstream docstrings contain LaTeX backslashes in ordinary strings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def function_rst(node: ast.FunctionDef, path: str) -> str:
    signature = f"{node.name}({ast.unparse(node.args)})"
    docstring = ast.get_docstring(node) or ""
    body = str(NumpyDocstring(docstring))
    if node.name == "read_density_grid" and path == "dragons/nbody/io.py":
        # Astropy adds this notice at import time; retain it without importing.
        body = (
            ".. deprecated:: 0.2.1\n"
            "   The read_density_grid function is deprecated and may be removed "
            "in a future version. Use :py:func:`dragons.nbody.io.read_grid` instead.\n\n"
            + body
        )
    body = body.rstrip() + f"\n\n`Source <{source_link(path, node.lineno, node.end_lineno)}>`__\n"
    return f".. py:function:: {signature}\n\n{indent(body.rstrip(), '   ')}\n\n"


def regrid_rst(source: str) -> str:
    match = re.search(r'def regrid\([\s\S]*?\):\s*("""[\s\S]*?""")', source)
    if match is None:
        raise ValueError("Cannot locate the Cython regrid signature/docstring")
    docstring = inspect.cleandoc(ast.literal_eval(match.group(1)))
    # Correct the two malformed NumPy field headers; preserve their information.
    docstring = docstring.replace(
        "old_grid (np.ndarray[float32, ndim=3]) :  Grid to be resampled",
        "old_grid : np.ndarray[float32, ndim=3]\n    Grid to be resampled",
    ).replace(
        "n_cell (int) :  n cells per dimension of new grid",
        "n_cell : int\n    n cells per dimension of new grid",
    )
    body = str(NumpyDocstring(docstring))
    line = source[:match.start()].count("\n") + 1
    body = body.rstrip() + f"\n\n`Source <{source_link('dragons/munge/regrid.pyx', line, len(source.splitlines()))}>`__\n"
    return ".. py:function:: regrid(old_grid, n_cell)\n\n" + indent(body.rstrip(), "   ") + "\n\n"


def api_page(page: str, sources: dict[str, str]) -> str:
    result = f".. _dragons-{page}:\n\n" + heading(page)
    for module in PAGES[page]:
        if page == "meraxes" or page == "nbody":
            result += heading(module.removeprefix("dragons."), "-")
        result += f".. py:module:: {module}\n\n"
        path = source_path(module)
        source = sources[path]
        if module == "dragons.munge.regrid":
            result += regrid_rst(source)
            continue
        tree = parse_module(source)
        docstring = ast.get_docstring(tree)
        if docstring:
            result += docstring + "\n\n"
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name in MEMBERS[module]:
            result += function_rst(functions[name], path)
        if not MEMBERS[module]:
            result += f"`Module source <{source_link(path)}>`__\n\n"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Offline DRAGONS checkout at the pinned revision")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "docs" / "post-processing"
    out.mkdir(parents=True, exist_ok=True)
    paths = ["README.rst", "AUTHORS.rst", "LICENSE", "docs/installation.rst"]
    paths.extend(source_path(module) for module in MEMBERS)

    def read(path: str) -> tuple[str, str]:
        if args.source_root:
            return path, (args.source_root / path).read_text(encoding="utf-8")
        with urlopen(f"{RAW}/{path}", timeout=60) as response:
            return path, response.read().decode("utf-8")

    with ThreadPoolExecutor(max_workers=8) as executor:
        sources = dict(executor.map(read, paths))

    readme = sources["README.rst"]
    for page in PAGES:
        readme = readme.replace(f":ref:`{page}`", f":ref:`{page} <dragons-{page}>`")
    (out / "readme.rst").write_text(readme, encoding="utf-8")
    installation = sources["docs/installation.rst"].replace(
        "https://github.com/smutch/dragons.git", REPOSITORY + ".git"
    )
    (out / "installation.rst").write_text(installation, encoding="utf-8")
    credits = sources["AUTHORS.rst"].rstrip() + "\n\n" + heading("License", "-")
    credits += (
        "DRAGONS Tools documentation © 2013 Simon Mutch and contributors.\n"
        "Distributed under the :download:`GNU General Public License v3 "
        "<../_static/DRAGONS-LICENSE.txt>`.\n\n"
        f"`DRAGONS Tools source <{REPOSITORY}>`__ · "
        "`Original documentation <https://meraxes-devs.github.io/dragons/>`__\n"
    )
    (out / "credits.rst").write_text(credits, encoding="utf-8")
    (root / "docs" / "_static" / "DRAGONS-LICENSE.txt").write_text(sources["LICENSE"], encoding="utf-8")
    for page in PAGES:
        (out / f"{page}.rst").write_text(api_page(page, sources), encoding="utf-8")

    print(f"Imported 3 narrative pages and {sum(map(len, MEMBERS.values()))} API functions in 4 pages.")


if __name__ == "__main__":
    main()
