from pathlib import Path

import nibabel as nib
import numpy as np


subjects = [
    "sub-09113", "sub-09210", "sub-09260", "sub-09261",
    "sub-09301", "sub-09380", "sub-09548", "sub-09587",
    "sub-09607",
]

masks = []

for subject in subjects:
    path = (
        Path("derivatives")
        / subject
        / "qc"
        / "mni"
        / "bold_brain_mask_space-MNI.nii.gz"
    )

    img = nib.load(path)
    masks.append(np.asarray(img.dataobj) > 0.5)

stack = np.stack(masks)

coverage = np.sum(stack, axis=0)
common = coverage == len(subjects)

out_dir = Path("derivatives/group/qc/ants")
out_dir.mkdir(parents=True, exist_ok=True)

reference = nib.load(
    Path("derivatives")
    / subjects[0]
    / "qc"
    / "mni"
    / "bold_brain_mask_space-MNI.nii.gz"
)

coverage_img = nib.Nifti1Image(
    coverage.astype(np.uint8),
    reference.affine,
    reference.header,
)

common_img = nib.Nifti1Image(
    common.astype(np.uint8),
    reference.affine,
    reference.header,
)

nib.save(
    coverage_img,
    out_dir / "coverage_count.nii.gz",
)

nib.save(
    common_img,
    out_dir / "common_mask.nii.gz",
)

print("Subjects:", len(subjects))
print("Common mask voxels:", np.count_nonzero(common))