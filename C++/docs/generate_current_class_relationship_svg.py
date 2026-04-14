#!/usr/bin/env python3

from pathlib import Path
from xml.sax.saxutils import escape


WIDTH = 3508
HEIGHT = 2480
BOX_W = 440
BOX_H = 170
RADIUS = 20
FONT = "Helvetica"


class Box:
    def __init__(self, key, title, lines, x, y, header_fill, body_fill="#fffdf8"):
        self.key = key
        self.title = title
        self.lines = lines
        self.x = x
        self.y = y
        self.header_fill = header_fill
        self.body_fill = body_fill

    @property
    def cx(self):
        return self.x + BOX_W / 2

    @property
    def cy(self):
        return self.y + BOX_H / 2


def rect(x, y, w, h, fill, stroke="#213547", stroke_width=3, radius=RADIUS):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" ry="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def text(x, y, value, size=28, weight="400", fill="#102027", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def line(x1, y1, x2, y2, stroke="#314e52", width=5, dash=None, marker_end=None):
    attrs = [
        f'x1="{x1}"',
        f'y1="{y1}"',
        f'x2="{x2}"',
        f'y2="{y2}"',
        f'stroke="{stroke}"',
        f'stroke-width="{width}"',
        'fill="none"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if marker_end:
        attrs.append(f'marker-end="url(#{marker_end})"')
    return f"<line {' '.join(attrs)}/>"


def label(x, y, value):
    return text(x, y, value, size=24, weight="600", fill="#475d63", anchor="middle")


def draw_box(box):
    header_h = 48
    lines = [
        rect(box.x, box.y, BOX_W, BOX_H, box.body_fill),
        rect(box.x + 1.5, box.y + 1.5, BOX_W - 3, header_h, box.header_fill, stroke="none", stroke_width=0, radius=18),
        f'<path d="M {box.x} {box.y + header_h} H {box.x + BOX_W}" stroke="#213547" stroke-width="3"/>',
        text(box.x + BOX_W / 2, box.y + 33, box.title, size=30, weight="700", fill="#102027", anchor="middle"),
    ]

    start_y = box.y + 82
    for idx, item in enumerate(box.lines):
        lines.append(text(box.x + 24, start_y + idx * 28, item, size=23, fill="#263238"))
    return "\n".join(lines)


def cluster(x, y, w, h, title, fill):
    return "\n".join(
        [
            rect(x, y, w, h, fill=fill, stroke="#7a8b8f", stroke_width=2, radius=26),
            text(x + 24, y + 42, title, size=34, weight="700", fill="#243238"),
        ]
    )


def main():
    root = Path(__file__).resolve().parent
    out = root / "current_class_relationships_a4.svg"

    boxes = {
        "Cell": Box("Cell", "Cell", ["abstract interface", "draw()", "drawOutline()"], 120, 1660, "#ffc857"),
        "Bacilli": Box("Bacilli", "Bacilli", ["legacy subtype", "inherits Cell"], 120, 1960, "#ffd9a8"),
        "Sphere": Box("Sphere", "Sphere", ["standalone model", "uses SphereParams"], 620, 1660, "#f4a261"),
        "Spheroid": Box("Spheroid", "Spheroid", ["primary tracked cell", "Frame stores vector<Spheroid>"], 620, 1960, "#e76f51"),
        "CellParams": Box("CellParams", "CellParams", ["base parameter record", "name"], 1120, 1460, "#bde0fe"),
        "SphereParams": Box("SphereParams", "SphereParams", ["x, y, z", "radius"], 1120, 1660, "#cfe8ff"),
        "SpheroidParams": Box("SpheroidParams", "SpheroidParams", ["x, y, z", "major/minor radius", "theta_x/y/z"], 1120, 1960, "#cfe8ff"),
        "CellConfig": Box("CellConfig", "CellConfig", ["abstract config base", "explodeConfig()"], 1660, 260, "#90e0ef"),
        "SphereConfig": Box("SphereConfig", "SphereConfig", ["x, y, z, radius", "PerturbParams"], 1660, 560, "#ade8f4"),
        "SpheroidConfig": Box("SpheroidConfig", "SpheroidConfig", ["x, y, z", "radii + theta params"], 1660, 860, "#ade8f4"),
        "PerturbParams": Box("PerturbParams", "PerturbParams", ["prob", "mu", "sigma"], 1660, 1160, "#caf0f8"),
        "SimulationConfig": Box("SimulationConfig", "SimulationConfig", ["iterations_per_cell", "z_scaling", "z_slices"], 2200, 410, "#95d5b2"),
        "ProbabilityConfig": Box("ProbabilityConfig", "ProbabilityConfig", ["perturbation", "split", "split_cost"], 2200, 710, "#95d5b2"),
        "BaseConfig": Box("BaseConfig", "BaseConfig", ["cellType", "cell*", "simulation", "prob"], 2200, 1010, "#74c69d"),
        "CellFactory": Box("CellFactory", "CellFactory", ["reads BaseConfig", "creates initial Spheroids"], 2740, 410, "#ffcad4"),
        "Frame": Box("Frame", "Frame", ["owns cells", "renders synthetic stack", "perturb()/trySplitCell()"], 2740, 960, "#f4acb7"),
        "Lineage": Box("Lineage", "Lineage", ["owns frames", "optimizes and copies cells"], 2740, 1510, "#f4acb7"),
        "LineageViewer": Box("LineageViewer", "LineageViewer", ["consumes frame snapshots", "visualizes lineage state"], 2740, 2010, "#ffe5ec"),
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#f8fbff"/>',
        '<stop offset="100%" stop-color="#fff7ed"/>',
        "</linearGradient>",
        '<marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="5" orient="auto" markerUnits="strokeWidth">',
        '<path d="M 0 0 L 12 5 L 0 10 z" fill="#314e52"/>',
        "</marker>",
        '<marker id="inherit" markerWidth="18" markerHeight="18" refX="14" refY="8" orient="auto" markerUnits="strokeWidth">',
        '<path d="M 0 8 L 14 0 L 14 16 z" fill="#ffffff" stroke="#314e52" stroke-width="1.5"/>',
        "</marker>",
        "</defs>",
        rect(0, 0, WIDTH, HEIGHT, "url(#bg)", stroke="none", stroke_width=0, radius=0),
        text(120, 120, "CellUniverse C++ Current Class Relationships", size=52, weight="700"),
        text(
            120,
            170,
            "Focused on actual class ownership, inheritance, and configuration dependencies in the current codebase.",
            size=28,
            fill="#4f5d75",
        ),
        cluster(60, 1400, 1550, 980, "Cell Models", "#fff3e0"),
        cluster(1600, 200, 1120, 1220, "Configuration", "#ecfeff"),
        cluster(2680, 200, 760, 2140, "Core Pipeline", "#fff1f2"),
    ]

    parts.extend(draw_box(box) for box in boxes.values())

    # Inheritance
    parts.extend(
        [
            line(boxes["Bacilli"].cx, boxes["Bacilli"].y, boxes["Cell"].cx, boxes["Cell"].y + BOX_H, marker_end="inherit"),
            line(boxes["SphereParams"].x, boxes["SphereParams"].cy, boxes["CellParams"].x + BOX_W, boxes["CellParams"].cy, marker_end="inherit"),
            line(boxes["SpheroidParams"].x, boxes["SpheroidParams"].cy, boxes["CellParams"].x + BOX_W, boxes["CellParams"].cy + 40, marker_end="inherit"),
            line(boxes["SphereConfig"].cx, boxes["SphereConfig"].y, boxes["CellConfig"].cx, boxes["CellConfig"].y + BOX_H, marker_end="inherit"),
            line(boxes["SpheroidConfig"].cx, boxes["SpheroidConfig"].y, boxes["CellConfig"].cx + 80, boxes["CellConfig"].y + BOX_H, marker_end="inherit"),
        ]
    )
    parts.extend(
        [
            label(300, 1810, "inherits"),
            label(1480, 1660, "inherits"),
            label(1480, 2020, "inherits"),
            label(1880, 505, "inherits"),
            label(1950, 805, "inherits"),
        ]
    )

    # Uses / composition
    parts.extend(
        [
            line(boxes["Sphere"].x + BOX_W, boxes["Sphere"].cy, boxes["SphereParams"].x, boxes["SphereParams"].cy, dash="12 10", marker_end="arrow"),
            line(boxes["Spheroid"].x + BOX_W, boxes["Spheroid"].cy, boxes["SpheroidParams"].x, boxes["SpheroidParams"].cy, dash="12 10", marker_end="arrow"),
            line(boxes["SphereConfig"].cx, boxes["SphereConfig"].y + BOX_H, boxes["PerturbParams"].cx, boxes["PerturbParams"].y, marker_end="arrow"),
            line(boxes["SpheroidConfig"].cx, boxes["SpheroidConfig"].y + BOX_H, boxes["PerturbParams"].cx + 90, boxes["PerturbParams"].y, marker_end="arrow"),
            line(boxes["BaseConfig"].x, boxes["BaseConfig"].cy - 70, boxes["SimulationConfig"].x + BOX_W, boxes["SimulationConfig"].cy, marker_end="arrow"),
            line(boxes["BaseConfig"].x, boxes["BaseConfig"].cy, boxes["ProbabilityConfig"].x + BOX_W, boxes["ProbabilityConfig"].cy, marker_end="arrow"),
            line(boxes["BaseConfig"].cx - 80, boxes["BaseConfig"].y, boxes["SpheroidConfig"].cx + 80, boxes["SpheroidConfig"].y + BOX_H, marker_end="arrow"),
            line(boxes["CellFactory"].x, boxes["CellFactory"].cy, boxes["BaseConfig"].x + BOX_W, boxes["BaseConfig"].cy - 40, dash="12 10", marker_end="arrow"),
            line(boxes["CellFactory"].x, boxes["CellFactory"].cy + 30, boxes["Spheroid"].x + BOX_W, boxes["Spheroid"].cy - 40, dash="12 10", marker_end="arrow"),
            line(boxes["Frame"].x, boxes["Frame"].cy - 40, boxes["SimulationConfig"].x + BOX_W, boxes["SimulationConfig"].cy + 20, dash="12 10", marker_end="arrow"),
            line(boxes["Frame"].x, boxes["Frame"].cy + 10, boxes["Spheroid"].x + BOX_W, boxes["Spheroid"].cy, marker_end="arrow"),
            line(boxes["Lineage"].x, boxes["Lineage"].cy - 60, boxes["BaseConfig"].x + BOX_W, boxes["BaseConfig"].cy + 60, marker_end="arrow"),
            line(boxes["Lineage"].x, boxes["Lineage"].cy, boxes["Frame"].x + BOX_W, boxes["Frame"].cy, marker_end="arrow"),
            line(boxes["Lineage"].x, boxes["Lineage"].cy + 70, boxes["Spheroid"].x + BOX_W, boxes["Spheroid"].cy + 20, dash="12 10", marker_end="arrow"),
            line(boxes["LineageViewer"].x, boxes["LineageViewer"].cy, boxes["Lineage"].x + BOX_W, boxes["Lineage"].cy + 120, dash="12 10", marker_end="arrow"),
        ]
    )
    parts.extend(
        [
            label(1370, 1700, "constructed from"),
            label(1370, 2000, "constructed from"),
            label(1880, 1310, "has many"),
            label(1960, 1400, "has many"),
            label(2360, 1000, "owns"),
            label(2360, 1090, "owns"),
            label(2030, 890, "points to"),
            label(2590, 1120, "reads"),
            label(2570, 1330, "creates"),
            label(2600, 1200, "renders with"),
            label(2610, 1500, "owns cells"),
            label(2580, 1700, "stores copy"),
            label(2600, 1780, "owns frames"),
            label(2590, 1890, "seeds / forwards"),
            label(3090, 1950, "visualizes"),
        ]
    )

    parts.extend(
        [
            text(
                120,
                2340,
                "Note: in the current headers, Spheroid and Sphere do not inherit from Cell; the active tracking path uses Spheroid directly.",
                size=26,
                weight="600",
                fill="#7c2d12",
            ),
            "</svg>",
        ]
    )

    out.write_text("\n".join(parts), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
