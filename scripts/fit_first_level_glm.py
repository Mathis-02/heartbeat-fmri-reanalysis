from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_stat_map
from nilearn.reporting import get_clusters_table

from inspect_bids import inspect_subject


def fit_glm(subject):
    info = inspect_subject(subject)
    derivatives = Path("derivatives") / subject

    bold_file = (
        derivatives / "func"
        / f"{subject}_task-heart_desc-stcMC_bold.nii.gz"
    )
    design_file = derivatives / "qc" / "design_matrix_stcMC.tsv"
    glm_mask = (
        derivatives / "qc" / "native"
        / "bold_brain_mask_stcMC.nii.gz"
    )

    stats_dir = derivatives / "stats" / "stcMC"
    qc_dir = derivatives / "qc" / "stcMC"

    stats_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    for path in (bold_file, design_file, glm_mask):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Load and check the design matrix
    design_matrix = pd.read_csv(design_file, sep="\t")
    bold = nib.load(bold_file)
    mask = nib.load(glm_mask)

    if bold.ndim != 4:
        raise ValueError("Expected a 4D BOLD image")

    if bold.shape[3] != info["n_volumes"]:
        raise ValueError("Unexpected number of BOLD volumes")

    if len(design_matrix) != bold.shape[3]:
        raise ValueError("Design matrix and BOLD have different lengths")

    if not np.isfinite(design_matrix.to_numpy()).all():
        raise ValueError("Design matrix contains non-finite values")

    if not np.allclose(bold.affine, mask.affine):
        raise ValueError("BOLD and mask affines differ")

    if bold.shape[:3] != mask.shape[:3]:
        raise ValueError("BOLD and mask have different spatial dimensions")

    # Fit the first-level model
    model = FirstLevelModel(
        t_r=info["tr"],
        noise_model="ar1",
        smoothing_fwhm=6.0,
        standardize=False,
        signal_scaling=0,
        mask_img=glm_mask,
        minimize_memory=True,
    )

    model = model.fit(
        str(bold_file),
        design_matrices=design_matrix,
    )

    mask_img = model.masker_.mask_img_
    mask_img.to_filename(qc_dir / "glm_mask_used.nii.gz")

    print("Mask voxels:", int(mask_img.get_fdata().sum()))

    # Heart - Sound contrast
    for column in ("Heart", "Sound"):
        if column not in design_matrix.columns:
            raise ValueError(f"Missing task regressor: {column}")

    contrast = np.zeros(len(design_matrix.columns))
    contrast[design_matrix.columns.get_loc("Heart")] = 1.0
    contrast[design_matrix.columns.get_loc("Sound")] = -1.0

    contrast_maps = model.compute_contrast(
        contrast,
        stat_type="t",
        output_type="all",
    )

    z_map = contrast_maps["z_score"]
    t_map = contrast_maps["stat"]
    effect_map = contrast_maps["effect_size"]

    # Save unthresholded statistical maps
    z_map.to_filename(stats_dir / "heart_minus_sound_z.nii.gz")
    t_map.to_filename(stats_dir / "heart_minus_sound_t.nii.gz")
    effect_map.to_filename(
        stats_dir / "heart_minus_sound_effect.nii.gz"
    )

    # Two-sided FDR correction
    thresholded_map, threshold = threshold_stats_img(
        z_map,
        alpha=0.05,
        height_control="fdr",
        cluster_threshold=10,
        two_sided=True,
    )

    thresholded_map.to_filename(
        stats_dir / "heart_minus_sound_z_fdr05.nii.gz"
    )

    print(f"FDR z threshold: {threshold:.3f}")

    clusters = get_clusters_table(
        z_map,
        stat_threshold=threshold,
        cluster_threshold=10,
        two_sided=True,
    )

    clusters.to_csv(
        stats_dir / "heart_minus_sound_clusters_fdr05.tsv",
        sep="\t",
        index=False,
    )

    # Quality-control figure
    display = plot_stat_map(
        thresholded_map,
        threshold=threshold,
        display_mode="ortho",
        title="Heart - Sound | FDR q < 0.05",
    )

    display.savefig(
        qc_dir / "heart_minus_sound_fdr05.png"
    )
    display.close()

    print(f"\nFirst-level GLM complete: {subject}")
    print("Statistical maps:", stats_dir)
    print("QC figure:", qc_dir / "heart_minus_sound_fdr05.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    args = parser.parse_args()

    fit_glm(args.subject)