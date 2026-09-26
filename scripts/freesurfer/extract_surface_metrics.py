from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nibabel.freesurfer import io


# FreeSurfer paths
subjects_dir = Path(os.environ["SUBJECTS_DIR"])
subject = "fsaverage"

subject_dir = subjects_dir / subject

white_path = subject_dir / "surf" / "lh.white"
pial_path = subject_dir / "surf" / "lh.pial"
thickness_path = subject_dir / "surf" / "lh.thickness"
annot_path = subject_dir / "label" / "lh.aparc.annot"


# Load surface data
white_vertices, white_faces = io.read_geometry(white_path)
pial_vertices, pial_faces = io.read_geometry(pial_path)
thickness = io.read_morph_data(thickness_path)
labels, _, region_names = io.read_annot(annot_path)

# Basic consistency checks
if not np.array_equal(white_faces, pial_faces):
    raise ValueError("White and pial surfaces have different topology.")

if len(white_vertices) != len(thickness):
    raise ValueError("Surface and thickness files have different vertex counts.")

if len(white_vertices) != len(labels):
    raise ValueError("Surface and annotation files have different vertex counts.")

print(f"Subject: {subject}")
print(f"Vertices: {len(white_vertices)}")
print(f"Faces: {len(white_faces)}")
print("Surface files are consistent.")

# Compute regional thickness statistics
region_stats = []

for region_id, region_name in enumerate(region_names):
    mask = labels == region_id
    region_thickness = thickness[mask]

    if region_thickness.size == 0:
        continue

    region_stats.append({
        "region": region_name.decode("utf-8"),
        "n_vertices": int(mask.sum()),
        "mean_thickness_mm": region_thickness.mean(),
        "sd_thickness_mm": region_thickness.std()
    })

stats_df = pd.DataFrame(region_stats)

# Remove the unlabeled vertices from the final results
stats_df = stats_df[stats_df["region"] != "unknown"]

print(f"Regions analyzed: {len(stats_df)}")

# Save results
output_dir = Path("results/freesurfer")
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "fsaverage_lh_aparc_thickness.csv"
stats_df.to_csv(csv_path, index=False)

print(f"Saved: {csv_path}")

# Plot mean cortical thickness by region
plot_df = stats_df.sort_values("mean_thickness_mm")

plt.figure(figsize=(10, 8))
plt.barh(plot_df["region"], plot_df["mean_thickness_mm"])
plt.xlabel("Mean cortical thickness (mm)")
plt.ylabel("Cortical region")
plt.title("Mean cortical thickness by region — fsaverage left hemisphere")
plt.tight_layout()

figure_path = output_dir / "fsaverage_lh_aparc_thickness.png"
plt.savefig(figure_path, dpi=300)
plt.close()

print(f"Saved: {figure_path}")