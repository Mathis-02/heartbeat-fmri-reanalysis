from pathlib import Path

import nibabel as nib
import numpy as np


SUBJECTS = [
    "sub-09113",
    "sub-09210",
    "sub-09260",
    "sub-09261",
    "sub-09301",
    "sub-09380",
    "sub-09381",
    "sub-09548",
    "sub-09587",
    "sub-09607",
]


def world_center(img):
    shape = np.array(img.shape[:3], dtype=float)
    voxel_center = (shape - 1) / 2

    return nib.affines.apply_affine(
        img.affine,
        voxel_center,
    )


for subject in SUBJECTS:

    t1_path = Path(
        f"data/ds003763/{subject}/anat/"
        f"{subject}_T1w.nii.gz"
    )

    epi_path = Path(
        f"derivatives/{subject}/func/"
        f"{subject}_task-heart_desc-stcMC_mean_bold.nii.gz"
    )

    t1 = nib.load(t1_path)
    epi = nib.load(epi_path)

    t1_center = world_center(t1)
    epi_center = world_center(epi)

    distance = np.linalg.norm(epi_center - t1_center)

    print(
        f"{subject}: "
        f"T1 center = {np.round(t1_center, 1)}, "
        f"EPI center = {np.round(epi_center, 1)}, "
        f"distance = {distance:.1f} mm"
    )