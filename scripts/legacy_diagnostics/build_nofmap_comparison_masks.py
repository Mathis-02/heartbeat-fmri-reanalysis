from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.masking import compute_epi_mask


SUBJECT = "sub-09210"

derivatives = Path("derivatives") / SUBJECT

bold_nofmap = (
    derivatives
    / "func"
    / f"{SUBJECT}_task-heart_desc-stcMCnofmap_space-MNI_bold.nii.gz"
)

anat_mask = (
    derivatives
    / "qc"
    / f"{SUBJECT}_T1w_brain_mask_space-MNI.nii.gz"
)

fmap_mask = (
    derivatives
    / "qc"
    / "stcMC"
    / "glm_mask_used.nii.gz"
)

out_dir = derivatives / "qc" / "stcMC_nofmap"
out_dir.mkdir(parents=True, exist_ok=True)

func_mask_file = out_dir / "bold_brain_mask_nofmap.nii.gz"
nofmap_mask_file = out_dir / "glm_mask_nofmap.nii.gz"
intersection_file = out_dir / "glm_mask_fmap_nofmap_intersection.nii.gz"


for path in [bold_nofmap, anat_mask, fmap_mask]:
    if not path.is_file():
        raise FileNotFoundError(path)


bold_img = nib.load(bold_nofmap)
anat_img = nib.load(anat_mask)
fmap_img = nib.load(fmap_mask)

# All masks must live on exactly the same MNI grid.
for name, img in [
    ("anatomical mask", anat_img),
    ("fieldmap GLM mask", fmap_img),
]:
    if img.shape != bold_img.shape[:3]:
        raise RuntimeError(
            f"{name} shape differs from nofmap BOLD grid"
        )

    if not np.allclose(img.affine, bold_img.affine):
        raise RuntimeError(
            f"{name} affine differs from nofmap BOLD grid"
        )


# Functional mask from the no-fieldmap BOLD.
func_mask_img = compute_epi_mask(bold_img)
nib.save(func_mask_img, func_mask_file)

func_mask = func_mask_img.get_fdata() > 0
anat_mask_data = anat_img.get_fdata() > 0
fmap_mask_data = fmap_img.get_fdata() > 0

# Same principle as our normal GLM mask:
# anatomical coverage AND functional coverage.
nofmap_mask = anat_mask_data & func_mask

intersection = fmap_mask_data & nofmap_mask


def save_binary_mask(data, reference, filename):
    img = nib.Nifti1Image(
        data.astype(np.uint8),
        reference.affine,
        reference.header,
    )
    nib.save(img, filename)


save_binary_mask(
    nofmap_mask,
    bold_img,
    nofmap_mask_file,
)

save_binary_mask(
    intersection,
    bold_img,
    intersection_file,
)


n_fmap = int(fmap_mask_data.sum())
n_nofmap = int(nofmap_mask.sum())
n_intersection = int(intersection.sum())

union = fmap_mask_data | nofmap_mask
n_union = int(union.sum())

dice = (
    2 * n_intersection
    / (n_fmap + n_nofmap)
)

jaccard = (
    n_intersection
    / n_union
)


print(f"Fieldmap mask voxels:     {n_fmap}")
print(f"No-fieldmap mask voxels:  {n_nofmap}")
print(f"Intersection voxels:      {n_intersection}")
print(f"Union voxels:             {n_union}")

print(f"\nDice coefficient:    {dice:.4f}")
print(f"Jaccard coefficient: {jaccard:.4f}")

print(
    "\nFmap-only voxels:   ",
    int((fmap_mask_data & ~nofmap_mask).sum()),
)

print(
    "Nofmap-only voxels: ",
    int((nofmap_mask & ~fmap_mask_data).sum()),
)

print(f"\nSaved: {nofmap_mask_file}")
print(f"Saved: {intersection_file}")