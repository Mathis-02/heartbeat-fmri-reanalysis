from pathlib import Path
import os
import subprocess


subjects = [
    "sub-09113", "sub-09210", "sub-09260", "sub-09261",
    "sub-09301", "sub-09380", "sub-09381", "sub-09548",
    "sub-09587", "sub-09607",
]

fsl_dir = Path(os.environ["FSLDIR"])
mni = fsl_dir / "data/standard/MNI152_T1_2mm.nii.gz"

for subject in subjects:

    print(f"\nProcessing {subject}")

    base = Path("derivatives") / subject
    reg = base / "reg"
    stats = base / "stats"

    effect = stats / "stcMC" / "heart_minus_sound_effect.nii.gz"
    out_dir = stats / "mni"
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = Path("data/ds003763") / subject / "anat" / f"{subject}_T1w.nii.gz"

    effect_t1 = out_dir / "heart_minus_sound_effect_space-T1.nii.gz"
    effect_mni = out_dir / "heart_minus_sound_effect_space-MNI.nii.gz"

    epi2t1_mat = reg / f"{subject}_stcMC_epi2t1.mat"
    epi2t1_warp = reg / f"{subject}_stcMC_epi2t1_warp.nii.gz"

    ants_dir = reg / "ants" / "final"
    ants_affine = ants_dir / f"{subject}_T1w_to_MNI_0GenericAffine.mat"
    ants_warp = ants_dir / f"{subject}_T1w_to_MNI_1Warp.nii.gz"

    # EPI -> T1
    if not effect_t1.exists():

        if epi2t1_warp.exists():
            subprocess.run([
                "applywarp",
                f"--in={effect}",
                f"--ref={t1}",
                f"--warp={epi2t1_warp}",
                f"--out={effect_t1}",
                "--interp=trilinear",
            ], check=True)

        else:
            subprocess.run([
                "flirt",
                "-in", str(effect),
                "-ref", str(t1),
                "-applyxfm",
                "-init", str(epi2t1_mat),
                "-interp", "trilinear",
                "-out", str(effect_t1),
            ], check=True)

    # T1 -> MNI
    if not effect_mni.exists():
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