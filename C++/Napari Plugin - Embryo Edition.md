see [the Original Napari Plugin](https://github.com/pnewstein/napari-3d-counter)  
see [the Embryo Adjusted Plugin](https://github.com/KaelynTaing/napari-3d-counter/tree/main)  
- adds notable features of Z Scale and Individual Point Size

Setup:
1) In main folder `napari-3d-counter`, run command `napari` in the terminal.
2) Add/drag tiff file 
3) In your top bar, you will see a `Plugins` tab. Go into `Plugins` -> `3D Counter` -> `Count3D` 

Notable Icons:
- **Top Left**: From Left to Right: 'x' to delete selected point, '+' to add a point, 'pointer icon' to select a point, 'four directional arrow icon' to move the entire file in space, 'box w/ arrow icon'' to drag and select
- **Bottom Left**: Square -> Cube Icon: toggle between 2D and 3D view
- **Bottom**: scroll bar in 2D to move through the z axis
- **Right**: From Top to Bottom: 'Individual Point Size' to Change a *<u>Selected</u>* Point Size, 'Z Scale' to Change the width of the 

Instructions on Usage:
Once run, you will see a `Point adder` layer, a default `Embryo` layer, and an `out of slice` layer.
1) Turn off `out of slice` layer with the eye icon near the layer in the bottom left
2) Add Points in the `Point adder` layer, in 2D. 
3) Switch to the `Embryo` layer, change the <u>blending</u> to opaque for visuals, and change to 3D.
4) Change the Z Scale, the points' positioning, and the points' size until you are satisfied.
5) Click `Save cells` to get the CSV!


Minor Bugs:
- `out of slice` layer "bugged" as it also changes size with what your "current" point size is. However, this layer is not much important, as it is used for keeping "copies" of the points. Turning this layer off is a way to bypass this for now, but this may be something that should be removed? It is in the original plugin.
- additionally, in this version, the `Individual Point Size` Slider does not work with multiple cell types, so the `DEFAULT_CONFIG` within `_widget.py` only starts with one cell type, "Embryo" in our case. 

Missing Features:
- Multiple Cell Layers for Different Types of Cells (and having individual point size work correctly with this)
- Removing cell_type in the CSV/Being able to Alter the Rows of the CSV
