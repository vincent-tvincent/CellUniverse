import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tifffile

stack = tifffile.imread('data/input/original_data/frame001.tif')
img = stack[14]
# Crop to a 200x200 window around the target cell for clarity
crop_y0, crop_x0 = 130, 50
img = img[crop_y0:crop_y0+200, crop_x0:crop_x0+200]

# Target = bright cell centroid (in cropped coords)
target = np.array([111.0 - crop_x0, 202.0 - crop_y0, 22.0, 18.0])
state  = np.array([target[0] + 50.0, target[1] - 45.0, 38.0, 32.0])
sigmas = np.array([2.5, 2.5, 1.5, 1.5])

snapshots = {0: state.copy()}
np.random.seed(0)
for i in range(1, 151):
    proposal = state + np.random.normal(0, 1, 4) * sigmas
    if np.sum((proposal - target)**2) < np.sum((state - target)**2):
        state = proposal
    if i in (50, 100, 150):
        snapshots[i] = state.copy()

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for ax, (it, s) in zip(axes.flat, snapshots.items()):
    ax.imshow(img, cmap='gray', vmin=0, vmax=img.max())
    ax.add_patch(Ellipse((s[0], s[1]), 2*s[2], 2*s[3],
                         edgecolor='red', facecolor='none', lw=2.5))
    ax.set_title(f'iteration {it}', fontsize=16)
    ax.axis('off')
plt.tight_layout()
plt.savefig('mc_demo.png', dpi=200, bbox_inches='tight')
print('saved mc_demo.png')
