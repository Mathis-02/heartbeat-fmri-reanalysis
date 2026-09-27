from pathlib import Path
import argparse
import os
import subprocess


def transform_contrast(subject, analysis):

    base = Path("derivatives") / subject
    reg = base / "reg"
    stats = base / "stats"

    effect = stats / analysis / "heart_minus_sound_effect.nii.gz"

    if analysis == "stcMC":
        out_dir = stats / "mni"
    else:
        out_dir = stats / f"{analysis}_mni"

    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = (
        Path("data/ds003763")
        / subject
        / "anat"
        / f"{subject}_T1w.nii.gz"
    )

    effect_t1 = (
        out_dir / "heart_minus_sound_effect_space-T1.nii.gz"
    )
    effect_mni = (
        out_dir / "heart_minus_sound_effect_space-MNI.nii.gz"
    )

    epi2t1_mat = reg / f"{subject}_stcMC_epi2t1.mat"
    epi2t1_warp = reg / f"{subject}_stcMC_epi2t1_warp.nii.gz"

    ants_dir = reg / "ants" / "final"
    ants_affine = (
        ants_dir / f"{subject}_T1w_to_MNI_0GenericAffine.mat"
    )
    ants_warp = (
        ants_dir / f"{subject}_T1w_to_MNI_1Warp.nii.gz"
    )

    fsl_dir = Path(os.environ["FSLDIR"])
    mni = fsl_dir / "data/standard/MNI152_T1_2mm.nii.gz"

    required = [
        effect,
        t1,
        ants_affine,
        ants_warp,
        mni,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    # EPI -> T1
    if epi2t1_warp.is_file():
        subprocess.run([
            "applywarp",
            f"--in={effect}",
            f"--ref={t1}",
            f"--warp={epi2t1_warp}",
            f"--out={effect_t1}",
            "--interp=trilinear",
        ], check=True)

    elif epi2t1_mat.is_file():
        subprocess.run([
            "flirt",
            "-in", str(effect),
            "-ref", str(t1),
            "-applyxfm",
            "-init", str(epi2t1_mat),
            "-interp", "trilinear",
            "-out", str(effect_t1),
        ], check=True)

    else:
        raise FileNotFoundError(
            f"No EPI-to-T1 transform found for {subject}"
        )

    # T1 -> MNI
    subprocess.run([
        "antsApplyTransforms",
        "-d", "3",
        "-i", str(effect_t1),
        "-r", str(mni),
        "-o", str(effect_mni),
        "-n", "Linear",
        "-t", str(ants_warp),
        "-t", str(ants_affine),
    ], check=True)

    print("Saved:", effect_mni)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    parser.add_argument(
        "--analysis",
        default="stcMC",
        choices=["stcMC", "stcMC_spikeFD05"],
    )
    args = parser.parse_args()

    transform_contrast(args.subject, args.analysis)
