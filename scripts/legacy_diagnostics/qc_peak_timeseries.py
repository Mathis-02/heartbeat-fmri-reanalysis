from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from nilearn.glm.first_level import FirstLevelModel


SUBJECT = "sub-09210"

BOLD_FILE = Path(
    f"derivatives/{SUBJECT}/func/"
    f"{SUBJECT}_task-heart_desc-mc_space-MNI_bold.nii.gz"
)

DESIGN_FILE = Path(
    f"derivatives/{SUBJECT}/qc/"
    "design_matrix.tsv"
)

GLM_MASK_FILE = Path(
    f"derivatives/{SUBJECT}/qc/"
    "glm_mask_final_subject.nii.gz"
)

QC_DIR = Path(
    f"derivatives/{SUBJECT}/qc"
)

QC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --- Peaks to inspect ---

peaks = {
    "Heart > Sound | main peak": (58.0, -16.0, 34.0),
    "Heart > Sound | cluster 2": (38.0, 54.0, 22.0),
    "Sound > Heart | main peak": (-28.0, 58.0, -10.0),
}


# --- Load design matrix ---

design_matrix = pd.read_csv(
    DESIGN_FILE,
    sep="\t",
)


# --- Load final GLM mask ---

glm_mask_img = nib.load(
    GLM_MASK_FILE
)

glm_mask_data = (
    glm_mask_img.get_fdata() > 0
)


# --- Build a tiny mask containing only the 3 peak voxels ---

peak_mask_data = np.zeros(
    glm_mask_data.shape,
    dtype=np.uint8,
)

peak_voxels = {}

inverse_affine = np.linalg.inv(
    glm_mask_img.affine
)

for name, coord in peaks.items():

    ijk_float = nib.affines.apply_affine(
        inverse_affine,
        coord,
    )

    ijk = tuple(
        np.round(ijk_float).astype(int)
    )

    if not glm_mask_data[ijk]:
        raise ValueError(
            f"{name}: voxel {ijk} at MNI {coord} "
            "is outside the final GLM mask."
        )

    peak_mask_data[ijk] = 1

    peak_voxels[name] = ijk

    print(
        f"{name}: "
        f"MNI {coord} -> voxel {ijk}"
    )


peak_mask_img = nib.Nifti1Image(
    peak_mask_data,
    affine=glm_mask_img.affine,
)

peak_mask_img.to_filename(
    QC_DIR / "glm_peak_qc_mask.nii.gz"
)


# --- Fit the same GLM, but only at the 3 selected voxels ---

qc_model = FirstLevelModel(
    t_r=2.0,
    noise_model="ar1",
    smoothing_fwhm=6.0,
    standardize=False,
    signal_scaling=0,
    mask_img=peak_mask_img,
    minimize_memory=False,
)

qc_model = qc_model.fit(
    str(BOLD_FILE),
    design_matrices=design_matrix,
)


# --- Retrieve predicted signal and residuals ---

predicted_img = qc_model.predicted_[0]
residuals_img = qc_model.residuals_[0]

predicted_data = predicted_img.get_fdata()
residuals_data = residuals_img.get_fdata()


# --- Extract the 3 time series ---

n_volumes = predicted_data.shape[3]
n_peaks = len(peaks)

predicted_ts = np.zeros(
    (n_volumes, n_peaks)
)

residual_ts = np.zeros(
    (n_volumes, n_peaks)
)

for i, name in enumerate(peaks):

    x, y, z = peak_voxels[name]

    predicted_ts[:, i] = (
        predicted_data[x, y, z, :]
    )

    residual_ts[:, i] = (
        residuals_data[x, y, z, :]
    )


# observed = predicted + residual

observed_ts = (
    predicted_ts + residual_ts
)

time = np.arange(
    n_volumes
) * 2.0

# --- Separate task and nuisance contributions ---

task_columns = [
    "Heart",
    "Heart_derivative",
    "Sound",
    "Sound_derivative",
]

task_betas = np.zeros(
    (len(task_columns), n_peaks)
)

for j, column in enumerate(task_columns):

    contrast = np.zeros(
        len(design_matrix.columns)
    )

    contrast[
        design_matrix.columns.get_loc(column)
    ] = 1.0

    beta_img = qc_model.compute_contrast(
        contrast,
        stat_type="t",
        output_type="effect_size",
    )

    beta_data = beta_img.get_fdata()

    for i, name in enumerate(peaks):

        x, y, z = peak_voxels[name]

        task_betas[j, i] = (
            beta_data[x, y, z]
        )

X_task = design_matrix[
    task_columns
].to_numpy()

task_predicted_ts = (
    X_task @ task_betas
)

nuisance_predicted_ts = (
    predicted_ts - task_predicted_ts
)
# --- Calculate R² at each peak ---

r2_values = []

for i in range(n_peaks):

    y = observed_ts[:, i]
    residuals = residual_ts[:, i]

    ss_res = np.sum(
        residuals ** 2
    )

    ss_tot = np.sum(
        (y - y.mean()) ** 2
    )

    r2 = 1 - (
        ss_res / ss_tot
    )

    r2_values.append(r2)


# --- Plot observed vs predicted and residuals ---

fig, axes = plt.subplots(
    n_peaks,
    2,
    figsize=(14, 9),
    sharex=True,
)

for i, (name, coord) in enumerate(
    peaks.items()
):

    axes[i, 0].plot(
        time,
        observed_ts[:, i],
        label="Observed",
        linewidth=1.0,
    )

    axes[i, 0].plot(
        time,
        predicted_ts[:, i],
        label="Predicted",
        linewidth=1.5,
    )

    axes[i, 0].set_title(
        f"{name}\n"
        f"MNI {coord} | R²={r2_values[i]:.3f}"
    )

    axes[i, 0].set_ylabel(
        "Signal (% baseline)"
    )

    axes[i, 0].legend()

    axes[i, 1].plot(
        time,
        residual_ts[:, i],
        linewidth=1.0,
    )

    axes[i, 1].axhline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    axes[i, 1].set_title(
        "Residuals"
    )

axes[-1, 0].set_xlabel(
    "Time (s)"
)

axes[-1, 1].set_xlabel(
    "Time (s)"
)

plt.tight_layout()

plt.savefig(
    QC_DIR / "glm_observed_vs_predicted.png",
    dpi=150,
)

plt.close()

# --- Plot task vs nuisance contributions ---

fig, axes = plt.subplots(
    n_peaks,
    2,
    figsize=(14, 9),
    sharex=True,
)

for i, (name, coord) in enumerate(
    peaks.items()
):

    # Task contribution
    axes[i, 0].plot(
        time,
        task_predicted_ts[:, i],
        linewidth=1.5,
    )

    axes[i, 0].axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axes[i, 0].set_title(
        f"{name}\n"
        f"MNI {coord} | Task contribution"
    )

    axes[i, 0].set_ylabel(
        "Predicted signal (% baseline)"
    )

    # Nuisance contribution
    axes[i, 1].plot(
        time,
        nuisance_predicted_ts[:, i],
        linewidth=1.5,
    )

    axes[i, 1].axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axes[i, 1].set_title(
        "Nuisance contribution"
    )

axes[-1, 0].set_xlabel(
    "Time (s)"
)

axes[-1, 1].set_xlabel(
    "Time (s)"
)

plt.tight_layout()

plt.savefig(
    QC_DIR / "glm_task_vs_nuisance.png",
    dpi=150,
)

plt.close()

# --- Print QC summary ---

print("\nR² values:")

for name, r2 in zip(
    peaks.keys(),
    r2_values,
):
    print(
        f"{name}: {r2:.3f}"
    )