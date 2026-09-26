
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

from inspect_bids import inspect_subject


def build_design_matrix(subject):
    info = inspect_subject(subject)

    derivatives = Path("derivatives") / subject

    bold_file = (
        derivatives / "func"
        / f"{subject}_task-heart_desc-stcMC_space-MNI_bold.nii.gz"
    )

    motion_file = (
        derivatives / "func"
        / f"{subject}_task-heart_desc-stcMC_bold.par"
    )

    events_file = Path(info["events"])

    qc_dir = derivatives / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    for path in (bold_file, motion_file, events_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    # BOLD acquisition timing
    bold = nib.load(bold_file)

    if bold.ndim != 4:
        raise ValueError("Expected a 4D BOLD image")

    n_volumes = bold.shape[3]
    tr = info["tr"]

    if n_volumes != info["n_volumes"]:
        raise ValueError("BOLD volume count differs from BIDS metadata")

    frame_times = np.arange(n_volumes) * tr + tr / 2

    # Experimental events
    events = pd.read_csv(events_file, sep="\t")

    required_columns = {"onset", "duration", "trial_type"}

    if not required_columns.issubset(events.columns):
        raise ValueError(
            f"Missing event columns: {required_columns - set(events.columns)}"
        )

    # Motion confounds
    motion = np.loadtxt(motion_file)
    motion = np.atleast_2d(motion)

    if motion.shape != (n_volumes, 6):
        raise ValueError(
            f"Unexpected motion parameters shape: {motion.shape}"
        )

    motion_df = pd.DataFrame(
        motion,
        columns=[
            "rot_x",
            "rot_y",
            "rot_z",
            "trans_x",
            "trans_y",
            "trans_z",
        ],
    )

    # First-level design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model="glover + derivative",
        drift_model="cosine",
        high_pass=0.01,
        add_regs=motion_df.values,
        add_reg_names=motion_df.columns.tolist(),
    )

    if not np.isfinite(design_matrix.to_numpy()).all():
        raise ValueError("Design matrix contains non-finite values")

    rank = np.linalg.matrix_rank(design_matrix.to_numpy())

    if rank < design_matrix.shape[1]:
        raise ValueError(
            f"Rank-deficient design matrix: "
            f"{rank}/{design_matrix.shape[1]}"
        )

    # Save design matrix
    design_file = qc_dir / "design_matrix_stcMC.tsv"

    design_matrix.to_csv(
        design_file,
        sep="\t",
        index=False,
    )

    # Design matrix figure
    plot_design_matrix(design_matrix)

    plt.savefig(
        qc_dir / "design_matrix_stcMC.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Collinearity checks
    X = design_matrix.drop(columns=["constant"])

    standard_deviations = X.std(ddof=0)

    if (standard_deviations == 0).any():
        raise ValueError(
            "Some non-constant regressors have zero variance"
        )

    X_standardized = (
        X - X.mean()
    ) / standard_deviations

    condition_number = np.linalg.cond(
        X_standardized.to_numpy()
    )

    corr = X.corr().abs()
    corr = corr.mask(np.eye(len(corr), dtype=bool))

    max_corr = corr.stack().sort_values(ascending=False)

    task_cols = [
        "Heart",
        "Heart_derivative",
        "Sound",
        "Sound_derivative",
    ]

    missing_task_cols = set(task_cols) - set(X.columns)

    if missing_task_cols:
        raise ValueError(
            f"Missing task regressors: {missing_task_cols}"
        )

    nuisance_cols = [
        col for col in X.columns
        if col not in task_cols
    ]

    task_nuisance_corr = (
        X[task_cols]
        .join(X[nuisance_cols])
        .corr()
        .loc[task_cols, nuisance_cols]
        .abs()
    )

    print(f"\nSubject: {subject}")
    print(f"Design matrix shape: {design_matrix.shape}")
    print(f"Matrix rank: {rank}/{design_matrix.shape[1]}")
    print(f"Standardized condition number: {condition_number:.2f}")

    print("\nHighest regressor correlations:")
    print(max_corr.head(10))

    print("\nHighest task-nuisance correlations:")
    print(
        task_nuisance_corr.stack()
        .sort_values(ascending=False)
        .head(10)
    )

    print(f"\nSaved design matrix: {design_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    args = parser.parse_args()

    build_design_matrix(args.subject)
