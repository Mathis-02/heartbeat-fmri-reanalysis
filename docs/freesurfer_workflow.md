# FreeSurfer notes

I used FreeSurfer 8.2 on Ubuntu 24.04 (WSL) to explore cortical surface data and learn how the main FreeSurfer outputs are organized.

For this part I worked with `fsaverage`, which is already reconstructed and distributed with FreeSurfer. I did not run `recon-all` on the subjects from ds003763 because of the computational cost on my laptop.

## Surface files

I first inspected the surfaces in Freeview, mainly:

- `lh.white` and `lh.pial`
- `lh.inflated`
- `lh.sphere`
- `lh.thickness`
- `lh.aparc.annot`

I also looked at `aseg.mgz` to understand the difference between volumetric segmentation and surface parcellation.

The left hemisphere of `fsaverage` has 163,842 vertices and 327,680 faces. `lh.white` and `lh.pial` use the same mesh topology, but the coordinates of the vertices are different.

## Reading the surfaces in Python

I then used NiBabel to read the FreeSurfer files directly in Python.

The script in `scripts/freesurfer/extract_surface_metrics.py` reads the white and pial meshes, cortical thickness values and the `aparc` labels. I added a few checks to make sure that the two surfaces have the same topology and that the number of vertices is consistent between the different files.

As a small example, the script groups the vertex-wise thickness values by `aparc` region and calculates the mean and standard deviation for each region.

The resulting table and figure are saved in `results/freesurfer/`.

This is only an exploration of precomputed FreeSurfer data. The next step would be to run a subject-specific reconstruction on suitable hardware and connect the cortical surfaces to the fMRI registration pipeline.