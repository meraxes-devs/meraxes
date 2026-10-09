"""Generate the guide's editable SVG diagrams with Python's standard library."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "_static"
COLORS = {
    "input": ("#427CC0", "#F1F6FC"),
    "galaxy": ("#BA7D25", "#FCF6EB"),
    "grid": ("#278470", "#EFF8F5"),
    "feedback": ("#8260A2", "#F6F1FA"),
    "output": ("#BD5864", "#FCF2F4"),
}


class Diagram:
    def __init__(self, width, height, title, description):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#203344} .heading{font-weight:700;font-size:17px} .body{font-size:14px} .label{font-size:13px}</style>',
            f'<rect width="{width}" height="{height}" fill="white"/>',
        ]
        self.nodes = {}

    def panel(self, x, y, width, height, label=None):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="#F5F7F9" stroke="#DCE2E7"/>')
        if label:
            self.label(x + 15, y + 25, label, anchor="start")

    def box(self, name, x, y, width, height, title, lines, kind="grid", optional=False):
        stroke, fill = COLORS[kind]
        dash = ' stroke-dasharray="6 4"' if optional else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>')
        self.parts.append(f'<text x="{x+width/2}" y="{y+25}" text-anchor="middle" class="heading">{escape(title)}</text>')
        for i, line in enumerate(lines):
            self.parts.append(f'<text x="{x+width/2}" y="{y+48+i*19}" text-anchor="middle" class="body">{escape(line)}</text>')
        self.nodes[name] = {
            "n": (x + width/2, y), "s": (x + width/2, y + height),
            "w": (x, y + height/2), "e": (x + width, y + height/2),
        }

    def label(self, x, y, text, anchor="middle", color=None):
        fill = f' style="fill:{color}"' if color else ""
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="label"{fill}>{escape(text)}</text>')

    def edge(self, source, target, source_port="s", target_port="n", via=(), color="#536676", dashed=False):
        points = [self.nodes[source][source_port], *via, self.nodes[target][target_port]]
        d = "M " + " L ".join(f"{x:g} {y:g}" for x, y in points)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.7" marker-end="url(#arrow)"{dash}/>')

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/name).write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


def workflow():
    d = Diagram(980, 1030, "Meraxes execution flow", "Launch, configure and initialize; then evolve halos, galaxies and radiation through snapshots before assembling the master file.")
    d.panel(205, 274, 758, 596, "Snapshot loop")
    stages = [
        ("launch", 18, 60, "Launch Meraxes", ["Initialize MPI"], "input"),
        ("params", 100, 70, "Read parameters and tables", ["Run settings, simulation data and physics tables"], "input"),
        ("init", 193, 64, "Initialize storage", ["Galaxy records and distributed grids"], "input"),
        ("halos", 310, 64, "Read halos", ["Update hosts; sample existing UVB feedback"], "input"),
        ("evolve", 398, 64, "Evolve galaxies", ["Gas, stars, black holes and mergers"], "galaxy"),
        ("prepare", 486, 64, "Construct source grids", ["Deposit galaxy radiation sources"], "grid"),
        ("igm", 574, 70, "Update the IGM", ["Heating, ionization and UVB history"], "grid"),
        ("tb", 668, 64, "Calculate 21-cm brightness", ["Neutral fraction, density and spin temperature"], "grid"),
        ("save", 786, 64, "Save outputs", ["Selected catalogues, grids and summaries"], "output"),
        ("master", 943, 64, "Assemble the master file", ["Link the saved snapshot products"], "output"),
    ]
    for name, y, height, title, lines, kind in stages:
        d.box(name, 265, y, 440, height, title, lines, kind)
    for a, b in zip(stages, stages[1:]):
        d.edge(a[0], b[0])
    d.box("density", 745, 486, 205, 64, "Simulation grids", ["Density; optional velocity"], "input")
    d.box("products", 745, 660, 205, 91, "Optional products", ["Power spectrum", "Lightcone"], "grid", True)
    d.edge("density", "prepare", "w", "e", color=COLORS["input"][0])
    d.edge("tb", "products", "e", "w", color=COLORS["grid"][0])
    d.edge("products", "save", "s", "e", via=[(847.5,818)], color=COLORS["grid"][0])
    d.edge("save", "halos", "w", "w", via=[(177,818),(177,342)], dashed=True)
    d.label(99, 598, "Next snapshot")
    d.label(605, 902, "After the last snapshot")
    d.save("workflow.svg")



def galaxy_physics():
    d = Diagram(1150, 700, "Galaxy physics and radiation feedback", "Gas supply and cooling feed star formation and black holes; stellar and UV background feedback couple the galaxy and intergalactic calculations.")
    d.box("halo", 35, 25, 315, 90, "Halo assembly", ["Growth, host changes and mergers", "Baryonic gas supply"], "input")
    d.box("gas", 35, 225, 315, 110, "Gas reservoirs", ["Infall; reincorporation", "Cooling: hot → cold gas", "Enrichment and ejection"], "galaxy")
    d.box("stars", 35, 465, 315, 100, "Star formation", ["Disk star formation and merger bursts", "Stellar populations and recycling"], "galaxy")
    d.box("bh", 435, 225, 285, 110, "Black holes and AGN", ["Hot- and cold-mode accretion", "Enabled radiative sources", "Radio-mode heating"], "galaxy")
    d.box("rad", 435, 465, 285, 100, "Escaped radiation", ["Ionizing stellar and AGN photons", "X-rays when enabled"], "grid")
    d.box("igm", 810, 465, 285, 100, "IGM evolution", ["Hydrogen ionization", "Thermal and radiation history"], "grid")
    d.edge("halo","gas")
    d.edge("gas","stars")
    d.edge("gas","bh","e","w",dashed=True)
    d.edge("stars","rad","e","w")
    d.edge("bh","rad")
    d.edge("rad","igm","e","w")
    d.edge("stars","gas","w","w",via=[(12,515),(12,280)],color=COLORS["feedback"][0],dashed=True)
    d.label(200, 400, "Stellar feedback and recycled material")
    d.edge("igm","gas","n","n",via=[(952.5,155),(192.5,155)],color=COLORS["feedback"][0],dashed=True)
    d.label(640, 145, "UVB history suppresses infall in later snapshots")
    d.label(560, 635, "Physical connections; execution order is given in the workflow diagram.")
    d.save("galaxy-physics.svg")



def output_tree():
    d = Diagram(1180, 690, "Meraxes output hierarchy", "The master file links snapshot groups. Each snapshot links rank galaxy catalogues, grid products and distribution-function summaries.")
    d.box("master", 440, 20, 300, 85, "meraxes.hdf5", ["Master assembled by rank 0", "After the snapshot loop"], "output")
    d.box("meta", 20, 20, 320, 105, "Master metadata", ["InputParams; Units", "HubbleConversions; gitdiff", "Attributes hold saved metadata"], "input")
    d.box("snaps", 845, 20, 315, 85, "Snapshot groups", ["SnapNNN", "Selected output snapshots"], "output")
    d.box("snap", 430, 205, 320, 95, "Snap", ["Redshift; LTTime; NGalaxies", "Links to the snapshot products"], "output")
    d.box("core", 20, 395, 320, 90, "Core<rank>", ["One group per writing MPI rank", "External links into rank files"], "output")
    d.box("grids", 430, 395, 320, 90, "Grids", ["External link to the grid file", "Distributed spatial products"], "grid")
    d.box("summaries", 840, 375, 320, 120, "Distribution functions", ["HMF; SMF; OIIILF; QuasarLF", "Optional UV and X-ray luminosity functions", "Direct links to rank-0 summaries"], "output")
    d.box("galdata", 20, 570, 320, 100, "Rank datasets", ["Galaxies: compound records", "Tree-index arrays when enabled", "Build-dependent fields"], "output")
    d.box("griddata", 430, 570, 320, 100, "Grid datasets and attributes", ["Source, thermal and ionization cubes", "21-cm arrays and scalar summaries", "Runtime-dependent products"], "grid")
    d.edge("master","meta","w","e")
    d.edge("master","snaps","e","w")
    d.edge("snaps","snap","s","n",via=[(1002.5,160),(590,160)])
    d.edge("snap","core","s","n",via=[(590,340),(180,340)])
    d.edge("snap","grids")
    d.edge("snap","summaries","s","n",via=[(590,340),(1000,340)])
    d.edge("core","galdata")
    d.edge("grids","griddata")
    d.save("snapshot-structure.svg")


if __name__ == "__main__":
    workflow()
    galaxy_physics()
    output_tree()
