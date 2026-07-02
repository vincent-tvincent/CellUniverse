"""Lineage helpers.

Cell names look like ``Cell type 1_20100``. The part after the last ``_`` is a
binary-tree lineage suffix (``20100``); the parent is the same suffix with its
last digit removed (``2010``). The prefix (``Cell type 1_``) is the root id and
is preserved. ``trash_*`` cells have no lineage parent.
"""

from __future__ import annotations

import re

_SUFFIX_RE = re.compile(r"^(.*_)(\d+)$")


def split_name(name: str) -> tuple[str, str] | None:
    """Return ``(prefix, suffix_digits)`` or None if the name has no numeric suffix."""
    m = _SUFFIX_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2)


def parent_of(name: str) -> str | None:
    """Return the parent cell name, or None for roots / trash / unparseable names.

    >>> parent_of("Cell type 1_20100")
    'Cell type 1_2010'
    >>> parent_of("Cell type 1_3") is None
    True
    >>> parent_of("trash_5") is None
    True
    """
    if name.startswith("trash") or "trash_" in name:
        return None
    parts = split_name(name)
    if parts is None:
        return None
    prefix, suffix = parts
    if len(suffix) <= 1:
        # A single-digit suffix is a lineage root; it has no split parent.
        return None
    return f"{prefix}{suffix[:-1]}"


def is_daughter_of(child: str, parent: str) -> bool:
    """True if ``child`` is a direct split daughter of ``parent``."""
    return parent_of(child) == parent


def committed_splits(prev_frame, curr_frame):
    """Detect splits that actually occurred between two frames.

    A *committed* split is authoritative (read from cells.csv), unlike a
    candidate_graph ``selected`` proposal which may never be applied: a cell
    present in ``prev_frame`` is absent in ``curr_frame`` and exactly its two
    lineage daughters have appeared.

    Returns a list of ``(parent_cell, daughter1_cell, daughter2_cell)`` where
    ``parent_cell`` is the prev-frame :class:`~cell_viz.model.Cell` (for the
    ghost) and the daughters are curr-frame Cells.
    """
    prev_by_name = {c.name: c for c in prev_frame.cells}
    curr_by_name = {c.name: c for c in curr_frame.cells}
    events = []
    seen_parents = set()
    for name, cell in curr_by_name.items():
        parent = parent_of(name)
        if parent is None or parent in seen_parents:
            continue
        if parent in prev_by_name and parent not in curr_by_name:
            daughters = sorted(
                (c for n, c in curr_by_name.items() if parent_of(n) == parent),
                key=lambda c: c.name,
            )
            if len(daughters) == 2:
                seen_parents.add(parent)
                events.append((prev_by_name[parent], daughters[0], daughters[1]))
    return events
