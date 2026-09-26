from pathlib import Path
import argparse
import subprocess
import sys
import numpy as np


from inspect_bids import inspect_subject
import json
import platform
import time
from datetime import datetime, timezone



RUN_LOG = None


def run(command):
    """Execute a command and record its result."""
    global RUN_LOG

    command = [str(arg) for arg in command]
    print("\n$", " ".join(command), flush=True)

    start = time.perf_counter()
    result = subprocess.run(command, check=False)
    duration = time.perf_counter() - start

    if RUN_LOG is not None:
        RUN_LOG["commands"].append({
            "command": command,
            "duration_seconds": round(duration, 2),
            "return_code": result.returncode,
        })

        save_run_log()

    result.check_returncode()


def save_run_log():
    if RUN_LOG is None:
        return

    log_file = Path(RUN_LOG["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_file.write_text(
        json.dumps(RUN_LOG, indent=2),
        encoding="utf-8",
    )


def preprocess_functional(subject):
    info = inspect_subject(subject)


    func_dir = Path("derivatives") / subject / "func"
    func_dir.mkdir(parents=True, exist_ok=True)

    stc = func_dir / (
        f"{subject}_task-heart_desc-stc_bold.nii.gz"
    )

    mc_prefix = func_dir / (
        f"{subject}_task-heart_desc-stcMC_bold"
    )

    mc_bold = Path(str(mc_prefix) + ".nii.gz")
    mc_par = Path(str(mc_prefix) + ".par")
    mc_mats = Path(str(mc_prefix) + ".mat")

    # 1. Slice-timing correction
    if stc.exists():
        print(f"Existing STC file: {stc}")
    else:
        run([
            sys.executable,
            "scripts/apply_slice_timing.py",
            subject,
        ])

    if not stc.is_file():
        raise RuntimeError("STC output was not created")


    # 2. Motion correction
    mc_outputs = [mc_bold, mc_par, mc_mats]

    if all(path.exists() for path in mc_outputs):
        print(f"Existing MCFLIRT outputs: {mc_prefix}")
    elif any(path.exists() for path in mc_outputs):
        raise RuntimeError(
            "Incomplete MCFLIRT outputs. Inspect the existing "
            "files before rerunning this stage."
        )
    else:
        run([
            "mcflirt",
            "-in", str(stc),
            "-out", str(mc_prefix),
            "-plots",
            "-mats",
            "-meanvol",
        ])

    if not all(path.exists() for path in mc_outputs):
        raise RuntimeError("MCFLIRT outputs are incomplete")

    # Generate the mean functional image for registration
    mean_bold = (
        func_dir / f"{subject}_task-heart_desc-stcMC_mean_bold.nii.gz"
    )

    if mean_bold.is_file():
        print(f"Existing mean BOLD: {mean_bold}")
    else:
        run([
            "fslmaths",
            str(mc_bold),
            "-Tmean",
            str(mean_bold),
        ])

    # 3. Motion quality control
    qc_dir = Path("derivatives") / subject / "qc/stcMC"
    fd_file = qc_dir / "framewise_displacement.tsv"

    if fd_file.exists():
        print(f"Existing motion QC: {fd_file}")
    else:
        run([
            sys.executable,
            "scripts/check_motion.py",
            subject,
        ])

    print(f"\nFunctional preprocessing complete: {subject}")
    print(f"Volumes expected: {info['n_volumes']}")
    print(f"TR: {info['tr']} s")



def preprocess_spatial(subject):
    import os
    import nibabel as nib

    info = inspect_subject(subject)

    derivatives = Path("derivatives") / subject
    func_dir = derivatives / "func"
    reg_dir = derivatives / "reg"

    bold = (
        func_dir
        / f"{subject}_task-heart_desc-stcMC_bold.nii.gz"
    )
    bold_mni = (
        func_dir
        / f"{subject}_task-heart_desc-stcMC_space-MNI_bold.nii.gz"
    )

    epi2t1 = reg_dir / f"{subject}_stcMC_epi2t1.mat"
    t1_to_mni = (
        reg_dir / "mni"
        / f"{subject}_T1w_to_MNI_affine.mat"
    )

    fsl_dir = os.environ.get("FSLDIR")
    if not fsl_dir:
        raise RuntimeError("FSLDIR is not defined")

    template = (
        Path(fsl_dir)
        / "data/standard/MNI152_T1_2mm.nii.gz"
    )

    if not bold.is_file():
        raise FileNotFoundError(bold)

    if not template.is_file():
        raise FileNotFoundError(template)

    # Register the functional and anatomical images if needed.
    if not epi2t1.is_file() or not t1_to_mni.is_file():
        run([
            sys.executable,
            "scripts/register_anatomy.py",
            subject,
        ])

    if not epi2t1.is_file() or not t1_to_mni.is_file():
        raise RuntimeError("Missing registration matrices")

    # Branch 1: fieldmap-based distortion correction.
    if info["fieldmap"] is not None:
        warp = (
            reg_dir
            / f"{subject}_stcMC_epi2t1_warp.nii.gz"
        )

        if not warp.is_file():
            run([
                sys.executable,
                "scripts/register_anatomy.py",
                subject,
            ])

        if not warp.is_file():
            raise FileNotFoundError(
                f"Missing distortion-correction warp: {warp}"
            )

        if bold_mni.is_file():
            print(f"Existing MNI BOLD (provenance to verify): {bold_mni}")
        else:
            run([
                "applywarp",
                f"--in={bold}",
                f"--ref={template}",
                f"--warp={warp}",
                f"--postmat={t1_to_mni}",
                f"--out={bold_mni}",
                "--interp=spline",
            ])

    # Branch 2: affine registration without fieldmap.
    else:
        epi2mni = (
            reg_dir
            / f"{subject}_stcMC_epi2mni.mat"
        )

        if epi2mni.is_file():
            print(f"Existing EPI-to-MNI matrix: {epi2mni}")
        else:
            run([
                "convert_xfm",
                "-omat", str(epi2mni),
                "-concat", str(t1_to_mni), str(epi2t1),
            ])

        if bold_mni.is_file():
            print(f"Existing MNI BOLD: {bold_mni}")
        else:
            run([
                "applywarp",
                f"--in={bold}",
                f"--ref={template}",
                f"--premat={epi2mni}",
                f"--out={bold_mni}",
                "--interp=spline",
            ])

    # Verify the output.
    image = nib.load(bold_mni)

    if image.ndim != 4:
        raise RuntimeError(
            f"Expected a 4D MNI BOLD image: {image.shape}"
        )

    if image.shape[3] != info["n_volumes"]:
        raise RuntimeError(
            f"Unexpected number of volumes: {image.shape}"
        )

    reference = nib.load(template)

    if image.shape[:3] != reference.shape[:3]:
        raise RuntimeError(
            "MNI BOLD and template have different dimensions"
        )

    if not np.allclose(image.affine, reference.affine):
        raise RuntimeError(
            "MNI BOLD and template have different affines"
        )

    print(f"\nSpatial preprocessing complete: {subject}")
    print(f"MNI BOLD shape: {image.shape}")


def run_first_level_analysis(subject):
    derivatives = Path("derivatives") / subject

    qc_dir = derivatives / "qc"
    stats_dir = derivatives / "stats" / "stcMC"

    mask_file = qc_dir / "glm_mask_final_subject_stcMC.nii.gz"
    design_file = qc_dir / "design_matrix_stcMC.tsv"

    expected_stats = [
        stats_dir / "heart_minus_sound_z.nii.gz",
        stats_dir / "heart_minus_sound_t.nii.gz",
        stats_dir / "heart_minus_sound_effect.nii.gz",
        stats_dir / "heart_minus_sound_z_fdr05.nii.gz",
        stats_dir / "heart_minus_sound_clusters_fdr05.tsv",
        qc_dir / "stcMC" / "heart_minus_sound_fdr05.png",
    ]

    # 1. Subject-specific GLM mask
    if mask_file.is_file():
        print(f"Existing GLM mask: {mask_file}")
    else:
        run([
            sys.executable,
            "scripts/build_glm_mask.py",
            subject,
        ])

    if not mask_file.is_file():
        raise RuntimeError("GLM mask was not created")

    # 2. First-level design matrix
    if design_file.is_file():
        print(f"Existing design matrix: {design_file}")
    else:
        run([
            sys.executable,
            "scripts/build_design_matrix.py",
            subject,
        ])

    if not design_file.is_file():
        raise RuntimeError("Design matrix was not created")

    # 3. First-level GLM
    if all(path.is_file() for path in expected_stats):
        print(f"Existing first-level GLM: {subject}")
    elif any(path.exists() for path in expected_stats):
        raise RuntimeError(
            "Incomplete GLM outputs. Inspect the existing files "
            "before rerunning the analysis."
        )
    else:
        run([
            sys.executable,
            "scripts/fit_first_level_glm.py",
            subject,
        ])

    if not all(path.is_file() for path in expected_stats):
        raise RuntimeError("First-level GLM outputs are incomplete")

    print(f"\nFirst-level analysis complete: {subject}")

def build_qc_summary(subject):
    import nibabel as nib
    import pandas as pd

    derivatives = Path("derivatives") / subject
    qc_dir = derivatives / "qc"

    fd_file = (
        qc_dir / "stcMC"
        / "framewise_displacement.tsv"
    )
    mask_file = (
        qc_dir
        / "glm_mask_final_subject_stcMC.nii.gz"
    )
    design_file = (
        qc_dir
        / "design_matrix_stcMC.tsv"
    )

    for path in (fd_file, mask_file, design_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Motion QC
    fd = pd.read_csv(
        fd_file,
        sep="\t",
    )["FD_mm"].to_numpy(dtype=float)

    if not np.isfinite(fd).all():
        raise ValueError("FD contains non-finite values")

    n_volumes = len(fd)
    high_motion = fd > 0.5
    n_high_motion = int(high_motion.sum())
    pct_high_motion = float(
        100 * high_motion.mean()
    )

    # GLM mask
    mask = nib.load(mask_file)
    mask_data = np.asarray(mask.dataobj)

    mask_voxels = int(
        np.count_nonzero(mask_data > 0)
    )

    # Design matrix
    design = pd.read_csv(
        design_file,
        sep="\t",
    )

    matrix = design.to_numpy(dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Design matrix contains non-finite values"
        )

    rank = int(np.linalg.matrix_rank(matrix))

    # Standardize columns before computing condition number,
    # excluding effectively constant columns.
    std = matrix.std(axis=0)
    nonconstant = std > 1e-12

    standardized = (
        matrix[:, nonconstant]
        - matrix[:, nonconstant].mean(axis=0)
    ) / std[nonconstant]

    condition_number = float(
        np.linalg.cond(standardized)
    )

    summary = {
        "subject": subject,
        "n_volumes": n_volumes,
        "motion": {
            "mean_fd_mm": float(np.mean(fd)),
            "median_fd_mm": float(np.median(fd)),
            "max_fd_mm": float(np.max(fd)),
            "fd_threshold_mm": 0.5,
            "n_fd_above_threshold": n_high_motion,
            "pct_fd_above_threshold": pct_high_motion,
        },
        "glm_mask_voxels": mask_voxels,
        "design": {
            "n_rows": int(design.shape[0]),
            "n_columns": int(design.shape[1]),
            "rank": rank,
            "condition_number_standardized": condition_number,
        },
    }

    output_file = qc_dir / "qc_summary.json"

    output_file.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSubject QC summary")
    print("-" * 50)
    print(f"Subject:              {subject}")
    print(f"Volumes:              {n_volumes}")
    print(f"Mean FD:              {np.mean(fd):.3f} mm")
    print(f"Median FD:            {np.median(fd):.3f} mm")
    print(f"Max FD:               {np.max(fd):.3f} mm")
    print(
        f"FD > 0.5 mm:          "
        f"{n_high_motion} / {n_volumes} "
        f"({pct_high_motion:.2f}%)"
    )
    print(f"GLM mask:             {mask_voxels} voxels")
    print(
        f"Design:               "
        f"{design.shape[0]} x {design.shape[1]}"
    )
    print(
        f"Design rank:          "
        f"{rank} / {design.shape[1]}"
    )
    print(
        f"Condition number:     "
        f"{condition_number:.2f}"
    )
    print(f"QC summary saved:     {output_file}")

def initialize_run_log(subject):
    global RUN_LOG

    info = inspect_subject(subject)

    log_dir = (
        Path("derivatives")
        / subject
        / "logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    log_file = log_dir / f"pipeline_{timestamp}.json"

    fsl_version = "unknown"
    fsl_dir = __import__("os").environ.get("FSLDIR")

    if fsl_dir:
        version_file = Path(fsl_dir) / "etc/fslversion"
        if version_file.is_file():
            fsl_version = version_file.read_text().strip()

    RUN_LOG = {
        "subject": subject,
        "started_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "python_version": platform.python_version(),
        "fsl_version": fsl_version,
        "fieldmap_detected": info["fieldmap"] is not None,
        "tr": info["tr"],
        "n_volumes": info["n_volumes"],
        "commands": [],
        "log_file": str(log_file),
    }

    save_run_log()

    print(f"Execution log: {log_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    args = parser.parse_args()

    initialize_run_log(args.subject)

    preprocess_functional(args.subject)
    preprocess_spatial(args.subject)
    run_first_level_analysis(args.subject)
    build_qc_summary(args.subject)

if __name__ == "__main__":
    main()