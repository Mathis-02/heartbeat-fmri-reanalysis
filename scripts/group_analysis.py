from pathlib import Path
from statsmodels.stats.multitest import fdrcorrection
import nibabel as nib
import numpy as np
from scipy.stats import ttest_1samp
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--analysis",
    choices=["main", "spike"],
    default="main",
)
args = parser.parse_args()

subjects = [
    "sub-09113", "sub-09210", "sub-09260", "sub-09261",
    "sub-09301", "sub-09380", "sub-09548", "sub-09587",
    "sub-09607",
]

mask_file = Path("derivatives/group/qc/ants/common_mask.nii.gz")
if args.analysis == "main":
    subject_stats_dir = "mni"
    out_dir = Path("derivatives/group/stats")
else:
    subject_stats_dir = "stcMC_spikeFD05_mni"
    out_dir = Path("derivatives/group/stats_spikeFD05")
out_dir.mkdir(parents=True, exist_ok=True)

mask_img = nib.load(mask_file)
mask = np.asarray(mask_img.dataobj) > 0

effects = []

for subject in subjects:
    path = (
        Path("derivatives")
        / subject
        / "stats"
        / subject_stats_dir
        / "heart_minus_sound_effect_space-MNI.nii.gz"
    )

    img = nib.load(path)

    if img.shape != mask_img.shape:
        raise ValueError(f"Shape mismatch for {subject}: {img.shape}")

    data = np.asarray(img.dataobj)
    effects.append(data[mask])

effects = np.stack(effects)

print("Subjects:", effects.shape[0])
print("Voxels:", effects.shape[1])

# Mean Heart - Sound effect
mean_values = effects.mean(axis=0)

# One-sample t-test against zero
t_values, p_values = ttest_1samp(
    effects,
    popmean=0,
    axis=0,
    nan_policy="omit",
)

# Put voxel values back into 3D volumes
mean_map = np.zeros(mask.shape, dtype=np.float32)
t_map = np.zeros(mask.shape, dtype=np.float32)
p_map = np.ones(mask.shape, dtype=np.float32)

mean_map[mask] = mean_values
t_map[mask] = t_values
p_map[mask] = p_values

header = mask_img.header.copy()
header.set_data_dtype(np.float32)

nib.save(
    nib.Nifti1Image(mean_map, mask_img.affine, header),
    out_dir / "heart_minus_sound_group_mean.nii.gz",
)

nib.save(
    nib.Nifti1Image(t_map, mask_img.affine, header),
    out_dir / "heart_minus_sound_group_t.nii.gz",
)

nib.save(
    nib.Nifti1Image(p_map, mask_img.affine, header),
    out_dir / "heart_minus_sound_group_p.nii.gz",
)

print("Mean effect range:",
      round(float(mean_values.min()), 4),
      "to",
      round(float(mean_values.max()), 4))

print("t range:",
      round(float(np.nanmin(t_values)), 3),
      "to",
      round(float(np.nanmax(t_values)), 3))

print("Uncorrected p < .05:",
      int(np.count_nonzero(p_values < 0.05)))

print("Saved to:", out_dir)

from statsmodels.stats.multitest import fdrcorrection

reject, p_fdr = fdrcorrection(
    p_values,
    alpha=0.05,
    method="indep",
)

t_fdr_map = np.zeros(mask.shape, dtype=np.float32)
t_fdr_map[mask] = np.where(reject, t_values, 0)

p_fdr_map = np.ones(mask.shape, dtype=np.float32)
p_fdr_map[mask] = p_fdr

n_fdr = int(np.count_nonzero(reject))

print("FDR q < .05:", n_fdr)

if n_fdr > 0:
    print(
        "FDR significant t range:",
        round(float(np.min(t_values[reject])), 3),
        "to",
        round(float(np.max(t_values[reject])), 3),
    )

nib.save(
    nib.Nifti1Image(t_fdr_map, mask_img.affine, header),
    out_dir / "heart_minus_sound_group_t_fdr05.nii.gz",
)

nib.save(
    nib.Nifti1Image(p_fdr_map, mask_img.affine, header),
    out_dir / "heart_minus_sound_group_p_fdr.nii.gz",
)