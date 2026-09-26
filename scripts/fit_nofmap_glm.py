from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from nilearn.glm.first_level import FirstLevelModel


SUBJECT = "sub-09210"

derivatives = Path("derivatives") / SUBJECT

bold_file = (
    derivatives
    / "func"
    / f"{SUBJECT}_task-heart_desc-stcMCnofmap_space-MNI_bold.nii.gz"
)

design_file = (
    derivatives
    / "qc"
    / "design_matrix_stcMC.tsv"
)

mask_file = (
    derivatives
    / "qc"
    / "stcMC_nofmap"
    / "glm_mask_fmap_nofmap_intersection.nii.gz"
)

stats_dir = derivatives / "stats" / "stcMC_nofmap"
stats_dir.mkdir(parents=True, exist_ok=True)


for path in [bold_file, design_file, mask_file]:
    if not path.is_file():
        raise FileNotFoundError(path)


bold_img = nib.load(bold_file)
mask_img = nib.load(mask_file)

design = pd.read_csv(
    design_file,
    sep="\t",
)


if bold_img.ndim != 4:
    raise RuntimeError(
        f"Expected 4D BOLD image, got {bold_img.shape}"
    )

if bold_img.shape[3] != len(design):
    raise RuntimeError(
        f"BOLD volumes ({bold_img.shape[3]}) "
        f"!= design rows ({len(design)})"
    )

if bold_img.shape[:3] != mask_img.shape:
    raise RuntimeError(
        "BOLD and mask dimensions differ"
    )

if not np.allclose(
    bold_img.affine,
    mask_img.affine,
):
    raise RuntimeError(
        "BOLD and mask affines differ"
    )


print(f"BOLD shape: {bold_img.shape}")
print(
    "Intersection mask voxels:",
    int((mask_img.get_fdata() > 0).sum()),
)


model = FirstLevelModel(
    t_r=2.0,
    noise_model="ar1",
    smoothing_fwhm=6.0,
    standardize=False,
    signal_scaling=0,
    mask_img=mask_img,
    minimize_memory=True,
)


print("\nFitting no-fieldmap GLM...")

model = model.fit(
    bold_img,
    design_matrices=design,
)


contrast = np.zeros(
    len(design.columns)
)

contrast[
    design.columns.get_loc("Heart")
] = 1.0

contrast[
    design.columns.get_loc("Sound")
] = -1.0


print("Computing Heart - Sound contrast...")

result = model.compute_contrast(
    contrast,
    stat_type="t",
    output_type="all",
)


outputs = {
    "z_score":
        stats_dir
        / "heart_minus_sound_z.nii.gz",

    "stat":
        stats_dir
        / "heart_minus_sound_t.nii.gz",

    "effect_size":
        stats_dir
        / "heart_minus_sound_effect.nii.gz",
}


for key, path in outputs.items():
    nib.save(
        result[key],
        path,
    )

    print(f"Saved: {path}")


print("\nNo-fieldmap GLM complete.")