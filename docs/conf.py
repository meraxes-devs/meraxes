project = "Meraxes Guide"
author = "Meraxes Guide contributors"

extensions = ["myst_parser", "sphinx.ext.mathjax", "sphinx_rtd_theme"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"

myst_enable_extensions = ["dollarmath", "amsmath"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_heading_anchors = 4
numfig = True
math_number_all = True
math_numfig = True
html_title = "Meraxes Guide"
html_css_files = ["guide.css"]
html_theme_options = {"navigation_depth": 3, "collapse_navigation": True}
html_context = {
    "display_github": True,
    "github_user": "meraxes-devs",
    "github_repo": "meraxes",
    "github_version": "master",
    "conf_py_path": "/docs/",
}

html_show_sourcelink = False
