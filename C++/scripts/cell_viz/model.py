"""Frame data model for the cell visualizer.

Pure data containers. No napari / numpy-heavy logic here so these can be
unit-tested without a GUI. Coordinates are kept in the tracker's native
``(x, y, z)`` space; conversion to napari ``(z, y, x)`` happens in geometry.py
using the z-scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cell:
    """A final fitted ellipsoid for one frame (one row of cells.csv)."""

    name: str
    x: float
    y: float
    z: float
    a_radius: float
    b_radius: float
    c_radius: float
    theta_x: float
    theta_y: float
    theta_z: float
    is_trash: bool
    brightness: float = 0.98  # synth fill value; from checkpoint when available

    @property
    def center(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def radii(self) -> tuple[float, float, float]:
        return (self.a_radius, self.b_radius, self.c_radius)

    @property
    def angles(self) -> tuple[float, float, float]:
        return (self.theta_x, self.theta_y, self.theta_z)


@dataclass(frozen=True)
class LumenCenter:
    """A CellLumen brightness-center candidate (candidate_graph kind=lumen_center)."""

    x: float
    y: float
    z: float
    voxels: float
    signal: float
    candidate_id: int | None = None

    @property
    def center(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class SplitEvent:
    """An accepted split (candidate_graph kind=split_pair, selected=1)."""

    parent: str
    d1: tuple[float, float, float]
    d2: tuple[float, float, float]
    score: float
    source: str = ""
    note: str = ""


@dataclass
class FrameData:
    """Everything needed to draw one frame."""

    frame: int
    cells: list[Cell] = field(default_factory=list)
    lumen_centers: list[LumenCenter] = field(default_factory=list)
    splits: list[SplitEvent] = field(default_factory=list)
    # Names in the order they were processed/picked (fit-order proxy).
    fit_order: list[str] = field(default_factory=list)

    def cell_by_name(self, name: str) -> Cell | None:
        for c in self.cells:
            if c.name == name:
                return c
        return None

    def cells_in_fit_order(self) -> list[Cell]:
        """Cells ordered by fit_order; any not listed are appended at the end."""
        index = {name: i for i, name in enumerate(self.fit_order)}
        big = len(self.fit_order)
        return sorted(self.cells, key=lambda c: index.get(c.name, big))
