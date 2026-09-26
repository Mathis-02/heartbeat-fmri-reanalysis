from pathlib import Path
import csv
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "ds003763"
DERIVATIVES = ROOT / "derivatives"

SUBJECTS = [
    "sub-09113",
    "sub-09210",
    "sub-09260",
    "sub-09261",
    "sub-09301",
    "sub-09380",
    "sub-09381",
    "sub-09548",
    "sub-09587",
    "sub-09607",
]


def run(command):
    print("\nRUN:")
    print(" ".join(map(str, command)))
    subprocess.run(command, check=True)


def get_mni_template():
    fsl_dir = os.environ.get("FSLDIR")

    if not fsl_dir:
        raise RuntimeError("FSLDIR is not defined.")

    template = Path(fsl_dir) / "data" / "standard" / "MNI152_T1_2mm.nii.gz"

    if not template.is_file():
        raise FileNotFoundError(f"MNI template not found: {template}")

    return template


def register_subject(subject):
    print(f"\n{'=' * 70}")
    print(f"ANTs registration: {subject}")
    print(f"{'=' * 70}")

    t1 = DATASET / subject / "anat" / f"{subject}_T1w.nii.gz"

    if not t1.is_file():
        raise FileNotFoundError(f"Missing T1w: {t1}")

    mni = get_mni_template()

    out_dir = DERIVATIVES / subject / "reg" / "ants" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = out_dir / f"{subject}_T1w_to_MNI_"

    warped = out_dir / f"{subject}_T1w_space-MNI.nii.gz"
    affine = out_dir / f"{subject}_T1w_to_MNI_0GenericAffine.mat"
    warp = out_dir / f"{subject}_T1w_to_MNI_1Warp.nii.gz"
    inverse_warp = out_dir / f"{subject}_T1w_to_MNI_1InverseWarp.nii.gz"
    jacobian = out_dir / f"{subject}_T1w_to_MNI_jacobian.nii.gz"

    expected_outputs = [
        warped,
        affine,
        warp,
        inverse_warp,
    ]

    if all(path.is_file() for path in expected_outputs):
        print("Existing ANTs registration found.")

    else:
        run([
            "antsRegistration",
            "--dimensionality", "3",
            "--float", "0",
            "--output", f"[{prefix},{warped}]",
            "--interpolation", "Linear",
            "--use-histogram-matching", "0",

            "--initial-moving-transform",
            f"[{mni},{t1},1]",

            "--transform", "Rigid[0.1]",
            "--metric",
            f"MI[{mni},{t1},1,32,Regular,0.25]",
            "--convergence",
            "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",

            "--transform", "Affine[0.1]",
            "--metric",
            f"MI[{mni},{t1},1,32,Regular,0.25]",
            "--convergence",
            "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",

            "--transform", "SyN[0.1,3,0]",
            "--metric",
            f"CC[{mni},{t1},1,4]",
            "--convergence",
            "[100x70x50x20,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
        ])

    if not warp.is_file():
        raise RuntimeError(f"Missing ANTs warp: {warp}")

    if not jacobian.is_file():
        run([
            "CreateJacobianDeterminantImage",
            "3",
            str(warp),
            str(jacobian),
            "0",
            "0",
        ])

    stats = subprocess.check_output([
        "fslstats",
        str(jacobian),
        "-R",
        "-M",
        "-S",
    ], text=True).strip().split()

    jac_min, jac_max, jac_mean, jac_sd = map(float, stats)

    print(
        f"Jacobian: min={jac_min:.4f}, "
        f"max={jac_max:.4f}, "
        f"mean={jac_mean:.4f}, "
        f"SD={jac_sd:.4f}"
    )

    return {
        "subject": subject,
        "jac_min": jac_min,
        "jac_max": jac_max,
        "jac_mean": jac_mean,
        "jac_sd": jac_sd,
    }


def write_qc_csv(results):
    qc_dir = DERIVATIVES / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    output = qc_dir / "ants_registration_qc.csv"

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "jac_min",
                "jac_max",
                "jac_mean",
                "jac_sd",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nQC table written to:")
    print(output)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python scripts/register_ants.py sub-XXXXX\n"
            "  python scripts/register_ants.py --all"
        )

    argument = sys.argv[1]

    if argument == "--all":
        results = []

        for subject in SUBJECTS:
            try:
                result = register_subject(subject)
                results.append(result)

            except Exception as error:
                print(f"\nERROR for {subject}: {error}")

        write_qc_csv(results)

    else:
        register_subject(argument)


if __name__ == "__main__":
    main()