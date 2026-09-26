from pathlib import Path
import argparse
import subprocess


def run(cmd):
    print("\n$", " ".join(str(x) for x in cmd))
    subprocess.run(
        [str(x) for x in cmd],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic EPI-to-T1 registration without fieldmap"
    )
    parser.add_argument(
        "subject",
        help="Subject ID, e.g. sub-09381",
    )
    args = parser.parse_args()

    subject = args.subject

    epi = Path(
        f"derivatives/{subject}/func/"
        f"{subject}_task-heart_desc-stcMC_mean_bold.nii.gz"
    )

    t1 = Path(
        f"data/ds003763/{subject}/anat/"
        f"{subject}_T1w.nii.gz"
    )

    t1_brain = Path(
        f"derivatives/{subject}/anat/"
        f"{subject}_T1w_brain.nii.gz"
    )

    out_dir = Path(
        f"derivatives/{subject}/reg/nofmap"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    output = out_dir / (
        f"{subject}_stcMC_epi2t1_nofmap"
    )

    # Basic checks
    for path in [epi, t1, t1_brain]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    run([
        "epi_reg",
        f"--epi={epi}",
        f"--t1={t1}",
        f"--t1brain={t1_brain}",
        f"--out={output}",
    ])

    registered = Path(
        str(output) + ".nii.gz"
    )

    if not registered.exists():
        raise RuntimeError(
            f"Registration output not created: {registered}"
        )

    print("\nNo-fieldmap registration complete.")
    print(f"Subject: {subject}")
    print(f"Registered EPI: {registered}")


if __name__ == "__main__":
    main()