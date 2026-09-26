from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting


SUBJECTS = ["sub-09113", "sub-09210"]

# Common threshold used ONLY for descriptive comparison/visualization.
COMMON_Z = 3.0

comparison_dir = Path("derivatives") / "group_qc" / "subject_comparison"
comparison_dir.mkdir(parents=True, exist_ok=True)

rows = []


for subject in SUBJECTS:

    stats_dir = (
        Path("derivatives")
        / subject
        / "stats"
        / "stcMC"
    )

    qc_dir = (
        Path("derivatives")
        / subject
        / "qc"
        / "stcMC"
    )

    z_file = stats_dir / "heart_minus_sound_z.nii.gz"
    effect_file = stats_dir / "heart_minus_sound_effect.nii.gz"
    mask_file = qc_dir / "glm_mask_used.nii.gz"

    for path in [z_file, effect_file, mask_file]:
        if not path.is_file():
            raise FileNotFoundError(path)

    z_img = nib.load(z_file)
    effect_img = nib.load(effect_file)
    mask_img = nib.load(mask_file)

    z = z_img.get_fdata()
    effect = effect_img.get_fdata()
    mask = mask_img.get_fdata() > 0

    if z.shape != mask.shape:
        raise RuntimeError(
            f"{subject}: Z map and mask shapes differ"
        )

    if effect.shape != mask.shape:
        raise RuntimeError(
            f"{subject}: effect map and mask shapes differ"
        )

    z_masked = z[mask]
    effect_masked = effect[mask]

    finite = np.isfinite(z_masked) & np.isfinite(effect_masked)

    z_masked = z_masked[finite]
    effect_masked = effect_masked[finite]

    n_mask = len(z_masked)

    positive = z_masked >= COMMON_Z
    negative = z_masked <= -COMMON_Z
    suprathreshold = np.abs(z_masked) >= COMMON_Z

    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    n_suprathreshold = int(suprathreshold.sum())

    rows.append({
        "subject": subject,
        "mask_voxels": n_mask,

        "z_mean": float(np.mean(z_masked)),
        "z_sd": float(np.std(z_masked)),
        "z_min": float(np.min(z_masked)),
        "z_max": float(np.max(z_masked)),

        "voxels_z_ge_3": n_positive,
        "voxels_z_le_minus3": n_negative,
        "voxels_abs_z_ge_3": n_suprathreshold,

        "percent_mask_z_ge_3":
            100 * n_positive / n_mask,

        "percent_mask_z_le_minus3":
            100 * n_negative / n_mask,

        "percent_mask_abs_z_ge_3":
            100 * n_suprathreshold / n_mask,

        "effect_mean":
            float(np.mean(effect_masked)),

        "effect_mean_z_ge_3":
            float(np.mean(effect_masked[positive]))
            if n_positive > 0 else np.nan,

        "effect_mean_z_le_minus3":
            float(np.mean(effect_masked[negative]))
            if n_negative > 0 else np.nan,

                "effect_sd":
            float(np.std(effect_masked)),

        "effect_median":
            float(np.median(effect_masked)),

        "effect_p05":
            float(np.percentile(effect_masked, 5)),

        "effect_p95":
            float(np.percentile(effect_masked, 95)),

        "effect_min":
            float(np.min(effect_masked)),

        "effect_max":
            float(np.max(effect_masked)),
    })

    output_png = (
        comparison_dir
        / f"{subject}_heart_minus_sound_Z_common3.png"
    )


    import matplotlib.pyplot as plt

    # Z distribution
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        z_masked,
        bins=100,
        density=True,
        alpha=0.8,
    )

    ax.axvline(0, linestyle="--")
    ax.axvline(COMMON_Z, linestyle=":")
    ax.axvline(-COMMON_Z, linestyle=":")

    ax.set_xlabel("Z")
    ax.set_ylabel("Density")
    ax.set_title(f"{subject} | Heart - Sound | Z distribution")

    fig.tight_layout()
    fig.savefig(
        comparison_dir / f"{subject}_z_distribution.png",
        dpi=150,
    )
    plt.close(fig)

    # Effect-size distribution
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        effect_masked,
        bins=100,
        density=True,
        alpha=0.8,
    )

    ax.axvline(0, linestyle="--")

    ax.set_xlabel("Contrast effect estimate")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{subject} | Heart - Sound | effect distribution"
    )

    fig.tight_layout()
    fig.savefig(
        comparison_dir / f"{subject}_effect_distribution.png",
        dpi=150,
    )
    plt.close(fig)

    display = plotting.plot_stat_map(
        z_img,
        threshold=COMMON_Z,
        display_mode="ortho",
        cut_coords=(0, 0, 0),
        symmetric_cbar=True,
        vmax=6,
        title=f"{subject} | Heart - Sound | common |Z| >= 3",
    )

    display.savefig(output_png, dpi=150)
    display.close()

    print(f"\n{subject}")
    print(f"Mask voxels: {n_mask}")
    print(f"Z >= 3: {n_positive}")
    print(f"Z <= -3: {n_negative}")
    print(f"|Z| >= 3: {n_suprathreshold}")
    print(f"Z range: {z_masked.min():.3f} to {z_masked.max():.3f}")


summary = pd.DataFrame(rows)

summary_file = comparison_dir / "heart_minus_sound_summary.tsv"

summary.to_csv(
    summary_file,
    sep="\t",
    index=False,
)

print("\nComparison summary:")
print(summary.to_string(index=False))

print(f"\nSaved: {summary_file}")