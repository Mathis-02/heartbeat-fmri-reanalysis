from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


SUBJECT = "sub-09210"
Z_THRESHOLD = 3.0

derivatives = Path("derivatives") / SUBJECT

fmap_dir = derivatives / "stats" / "stcMC"
nofmap_dir = derivatives / "stats" / "stcMC_nofmap"

mask_file = (
    derivatives
    / "qc"
    / "stcMC_nofmap"
    / "glm_mask_fmap_nofmap_intersection.nii.gz"
)

files = {
    "fmap_z": fmap_dir / "heart_minus_sound_z.nii.gz",
    "fmap_t": fmap_dir / "heart_minus_sound_t.nii.gz",
    "fmap_effect": fmap_dir / "heart_minus_sound_effect.nii.gz",

    "nofmap_z": nofmap_dir / "heart_minus_sound_z.nii.gz",
    "nofmap_t": nofmap_dir / "heart_minus_sound_t.nii.gz",
    "nofmap_effect": nofmap_dir / "heart_minus_sound_effect.nii.gz",
}

for path in [mask_file, *files.values()]:
    if not path.is_file():
        raise FileNotFoundError(path)


mask_img = nib.load(mask_file)
mask = mask_img.get_fdata() > 0

data = {}

for name, path in files.items():
    img = nib.load(path)

    if img.shape != mask.shape:
        raise RuntimeError(
            f"{name}: incompatible dimensions"
        )

    if not np.allclose(img.affine, mask_img.affine):
        raise RuntimeError(
            f"{name}: incompatible affine"
        )

    data[name] = img.get_fdata()[mask]


# Keep voxels finite in every map.
valid = np.ones(mask.sum(), dtype=bool)

for values in data.values():
    valid &= np.isfinite(values)

for name in data:
    data[name] = data[name][valid]


fmap_z = data["fmap_z"]
nofmap_z = data["nofmap_z"]

fmap_t = data["fmap_t"]
nofmap_t = data["nofmap_t"]

fmap_effect = data["fmap_effect"]
nofmap_effect = data["nofmap_effect"]


# Reconstruct contrast standard errors.
valid_fmap_t = np.abs(fmap_t) > 1e-8
valid_nofmap_t = np.abs(nofmap_t) > 1e-8
valid_se = valid_fmap_t & valid_nofmap_t

fmap_se = (
    np.abs(fmap_effect[valid_se])
    / np.abs(fmap_t[valid_se])
)

nofmap_se = (
    np.abs(nofmap_effect[valid_se])
    / np.abs(nofmap_t[valid_se])
)


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def mean_absolute_difference(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(
        np.sqrt(np.mean((a - b) ** 2))
    )


fmap_positive = fmap_z >= Z_THRESHOLD
nofmap_positive = nofmap_z >= Z_THRESHOLD

fmap_negative = fmap_z <= -Z_THRESHOLD
nofmap_negative = nofmap_z <= -Z_THRESHOLD

fmap_super = np.abs(fmap_z) >= Z_THRESHOLD
nofmap_super = np.abs(nofmap_z) >= Z_THRESHOLD

changed = fmap_super != nofmap_super


summary = {
    "voxels_compared": len(fmap_z),

    "effect_correlation":
        correlation(fmap_effect, nofmap_effect),

    "effect_mean_fmap":
        float(np.mean(fmap_effect)),

    "effect_mean_nofmap":
        float(np.mean(nofmap_effect)),

    "effect_mean_difference_fmap_minus_nofmap":
        float(np.mean(fmap_effect - nofmap_effect)),

    "effect_mean_absolute_difference":
        mean_absolute_difference(
            fmap_effect,
            nofmap_effect,
        ),

    "effect_rmse":
        rmse(fmap_effect, nofmap_effect),

    "z_correlation":
        correlation(fmap_z, nofmap_z),

    "z_mean_fmap":
        float(np.mean(fmap_z)),

    "z_mean_nofmap":
        float(np.mean(nofmap_z)),

    "z_mean_difference_fmap_minus_nofmap":
        float(np.mean(fmap_z - nofmap_z)),

    "z_mean_absolute_difference":
        mean_absolute_difference(
            fmap_z,
            nofmap_z,
        ),

    "z_rmse":
        rmse(fmap_z, nofmap_z),

    "se_mean_fmap":
        float(np.mean(fmap_se)),

    "se_mean_nofmap":
        float(np.mean(nofmap_se)),

    "se_median_fmap":
        float(np.median(fmap_se)),

    "se_median_nofmap":
        float(np.median(nofmap_se)),

    "positive_z3_fmap":
        int(fmap_positive.sum()),

    "positive_z3_nofmap":
        int(nofmap_positive.sum()),

    "negative_z3_fmap":
        int(fmap_negative.sum()),

    "negative_z3_nofmap":
        int(nofmap_negative.sum()),

    "abs_z3_fmap":
        int(fmap_super.sum()),

    "abs_z3_nofmap":
        int(nofmap_super.sum()),

    "threshold_classification_changed":
        int(changed.sum()),

    "threshold_classification_changed_percent":
        float(100 * changed.mean()),
}


print("\nFIELD MAP vs NO-FIELD MAP")
print("=" * 50)

for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")


output_dir = (
    Path("derivatives")
    / "group_qc"
    / "fieldmap_control"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

pd.DataFrame(
    [summary]
).to_csv(
    output_dir / "sub-09210_fmap_vs_nofmap.tsv",
    sep="\t",
    index=False,
)

print(
    "\nSaved:",
    output_dir / "sub-09210_fmap_vs_nofmap.tsv",
)