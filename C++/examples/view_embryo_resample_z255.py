import os
import argparse
import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import zoom
import napari
from qtpy.QtCore import QTimer

def load_tiff_volume(path):
    pages = imageio.mimread(path)
    zstack = []
    for p in pages:
        arr = np.asarray(p)
        if arr.ndim == 3:
            arr = arr[..., 0]
        zstack.append(arr)
    return np.stack(zstack, axis=0)

def resample_z(vol, target_z):
    z, y, x = vol.shape
    if z == target_z:
        return vol
    factor = target_z / float(z)
    out = zoom(vol, zoom=(factor, 1.0, 1.0), order=1)
    if out.shape[0] > target_z:
        out = out[:target_z, :, :]
    elif out.shape[0] < target_z:
        pad = target_z - out.shape[0]
        out = np.pad(out, ((0, pad), (0, 0), (0, 0)), mode="edge")
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", required=True)
ap.add_argument("--target_z", type=int, default=255)
ap.add_argument("--fps", type=int, default=5)
args = ap.parse_args()

paths = []
for i in range(0, 10):
    p = os.path.join(args.input_dir, f"t{i:03d}.tif")
    if not os.path.isfile(p):
        raise RuntimeError(f"Missing file: {p}")
    paths.append(p)

vols = []
for p in paths:
    v = load_tiff_volume(p).astype(np.float32)
    v = resample_z(v, args.target_z)
    vols.append(v)

data = np.stack(vols, axis=0)

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(data, name="embryo_resample_z", rendering="mip")

interval = int(1000 / max(1, args.fps))
timer = QTimer()
t_axis = 0
t_max = data.shape[0] - 1

def step():
    cur = int(viewer.dims.current_step[t_axis])
    nxt = cur + 1
    if nxt > t_max:
        nxt = 0
    viewer.dims.set_current_step(t_axis, nxt)

timer.timeout.connect(step)
timer.start(interval)

napari.run()
