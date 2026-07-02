"""Apply a Scene to a napari viewer (the only napari-touching module besides app).

Layers are created with ndim=3 up front: the data is always 3D ``(z, y, x)`` and
the viewer may display in 2D or 3D. Creating empty layers without ndim makes
napari assume ndim=2, which crashes in 3D display when 3D data is added later
(IndexError in the world->layer units scale).
"""

from __future__ import annotations

import numpy as np

from .scene import GHOST_COLOR, Scene

_GHOST_RGBA = list(GHOST_COLOR)


class NapariRenderer:
    """Owns the napari layers and rewrites them per rendered frame."""

    def __init__(self, viewer, raw_shape):
        self.viewer = viewer
        self.raw_shape = raw_shape
        self._raw = None
        self._synth = None
        self._init_layers()

    def _init_layers(self):
        v = self.viewer
        zeros = np.zeros(self.raw_shape, dtype="uint8")
        # Bottom-to-top: real (gray) -> synth (green, stacked) -> overlays.
        self._raw = v.add_image(zeros, name="real", colormap="gray")
        self._synth = v.add_image(
            zeros.copy(), name="synth", colormap="green", blending="additive",
            opacity=0.45, contrast_limits=(0, 255),
        )
        # Last-frame split parent: orange, thick, semi-transparent ("ghost").
        self.ghosts = v.add_shapes(
            name="parent_ghosts (f-1)", ndim=3, shape_type="path",
            edge_color="orange", edge_width=2.2, opacity=0.55,
        )
        self.links = v.add_shapes(
            name="split_links", ndim=3, shape_type="path", edge_color="orange",
            edge_width=1.0, opacity=0.7,
        )
        self.cells = v.add_shapes(
            name="cells", ndim=3, shape_type="path", edge_width=1.6, opacity=0.95,
        )
        self.lumen = v.add_points(
            name="lumen_centers", ndim=3, face_color="cyan", border_color="black",
            symbol="disc", size=6.0, opacity=0.9,
        )
        self.labels = v.add_points(
            name="labels", ndim=3, size=0.1, face_color="transparent",
            border_color="transparent",
        )
        self.ghost_labels = v.add_points(
            name="ghost_labels", ndim=3, size=0.1, face_color="transparent",
            border_color="transparent",
        )

    @staticmethod
    def _set_shapes(layer, paths, colors=None):
        # Clearing then adding is safe now that the layer is ndim=3.
        layer.data = []
        if not paths:
            return
        edge = colors if colors is not None else "white"
        layer.add(list(paths), shape_type="path", edge_color=edge)

    def set_raw(self, image: np.ndarray):
        self._raw.data = image

    def set_synth(self, image):
        """Stack the C++ synth volume on the real (green). None -> hide."""
        if image is None:
            self._synth.data = np.zeros(self.raw_shape, dtype="uint8")
            self._synth.visible = False
        else:
            self._synth.data = image
            self._synth.visible = True

    @staticmethod
    def _set_points_text(layer, points, texts, color, size=7):
        if points is not None and getattr(points, "size", 0):
            layer.data = points
            layer.text = {"string": list(texts), "size": size,
                          "color": color, "anchor": "center"}
        else:
            layer.data = np.empty((0, 3))

    def render(self, scene: Scene):
        ghost_cols = [_GHOST_RGBA] * len(scene.ghost_paths)
        link_cols = [_GHOST_RGBA] * len(scene.link_paths)
        self._set_shapes(self.ghosts, scene.ghost_paths, ghost_cols or None)
        self._set_shapes(self.links, scene.link_paths, link_cols or None)
        self._set_shapes(self.cells, scene.ring_paths, scene.ring_colors or None)

        if scene.lumen_points.size:
            self.lumen.data = scene.lumen_points
            self.lumen.size = scene.lumen_sizes
        else:
            self.lumen.data = np.empty((0, 3))

        self._set_points_text(self.labels, scene.label_points, scene.label_texts, "white")
        self._set_points_text(self.ghost_labels, scene.ghost_label_points,
                              scene.ghost_label_texts, "orange", size=8)
