from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


SUBJECTS = ["sub-09113", "sub-09210"]

output_dir = (
    Path("derivatives")
    / "group_qc"
    / "subject_comparison"
)

output_dir.mkdir(parents=True, exist_ok=True)

rows = []


for subject in SUBJECTS:

    stats_dir = (
        Path("derivatives")
        / subject
        / "stats"
        / "stcMC"
    )

    mask_file = (
        Path("derivatives")
        / subject
        / "qc"
        / "stcMC"
        / "glm_mask_used.nii.gz"
    )

    effect_file = stats_dir / "heart_minus_sound_effect.nii.gz"
    t_file = stats_dir / "heart_minus_sound_t.nii.gz"

    for path in [effect_file, t_file, mask_file]:
        if not path.is_file():
            raise FileNotFoundError(path)

    effect_img = nib.load(effect_file)
    t_img = nib.load(t_file)
    mask_img = nib.load(mask_file)

    effect = effect_img.get_fdata()
    t = t_img.get_fdata()
    mask = mask_img.get_fdata() > 0

    if effect.shape != t.shape or effect.shape != mask.shape:
        raise RuntimeError(
            f"{subject}: incompatible image dimensions"
        )

    # Avoid division by zero / numerically unstable values.
    valid = (
        mask
        & np.isfinite(effect)
        & np.isfinite(t)
        & (np.abs(t) > 1e-8)
    )

    se = np.full(effect.shape, np.nan)

    se[valid] = (
        np.abs(effect[valid])
        / np.abs(t[valid])
    )

    se_values = se[valid]

    # Remove any remaining non-finite values defensively.
    se_values = se_values[np.isfinite(se_values)]

    rows.append({
        "subject": subject,
        "valid_voxels": len(se_values),
        "se_mean": float(np.mean(se_values)),
        "se_median": float(np.median(se_values)),
        "se_sd": float(np.std(se_values)),
        "se_p05": float(np.percentile(se_values, 5)),
        "se_p25": float(np.percentile(se_values, 25)),
        "se_p75": float(np.percentile(se_values, 75)),
        "se_p95": float(np.percentile(se_values, 95)),
    })

    # Save the reconstructed SE image for later inspection.
    se_to_save = np.nan_to_num(
        se,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    se_img = nib.Nifti1Image(
        se_to_save.astype(np.float32),
        effect_img.affine,
        effect_img.header,
    )

    se_file = (
        output_dir
        / f"{subject}_heart_minus_sound_se.nii.gz"
    )

    nib.save(se_img, se_file)

    print(f"\n{subject}")
    print(f"Valid voxels: {len(se_values)}")
    print(f"SE mean:   {np.mean(se_values):.6f}")
    print(f"SE median: {np.median(se_values):.6f}")
    print(
        "SE 5-95%: "
        f"{np.percentile(se_values, 5):.6f} - "
        f"{np.percentile(se_values, 95):.6f}"
    )


summary = pd.DataFrame(rows)

summary_file = (
    output_dir
    / "heart_minus_sound_uncertainty_summary.tsv"
)

summary.to_csv(
    summary_file,
    sep="\t",
    index=False,
)

print("\nUncertainty comparison:")
print(summary.to_string(index=False))

print(f"\nSaved: {summary_file}")