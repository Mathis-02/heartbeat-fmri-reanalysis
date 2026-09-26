from pathlib import Path
import argparse
import subprocess

from inspect_bids import inspect_subject


def run(command):
    print("\n$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def prepare_fieldmap(subject):
    info = inspect_subject(subject)
    fieldmap = info["fieldmap"]

    if fieldmap is None:
        print(f"No fieldmap associated with {subject}")
        return

    fmap_dir = Path("derivatives") / subject / "fmap"
    fmap_dir.mkdir(parents=True, exist_ok=True)

    magnitude = Path(fieldmap["magnitude1"])
    phasediff = Path(fieldmap["phasediff"])

    magnitude_brain = (
        fmap_dir / f"{subject}_magnitude1_brain.nii.gz"
    )

    fieldmap_rads = (
        fmap_dir / f"{subject}_fieldmap_rads.nii.gz"
    )

    # FSL expects the echo-time difference in milliseconds.
    delta_te_ms = fieldmap["delta_te"] * 1000

    # Extract the brain from the magnitude image.
    if magnitude_brain.is_file():
        print(f"Existing brain-extracted magnitude: {magnitude_brain}")
    else:
        run([
            "bet",
            str(magnitude),
            str(magnitude_brain),
            "-f", "0.5",
            "-m",
        ])

    # Convert the Siemens phase-difference image to rad/s.
    if fieldmap_rads.is_file():
        print(f"Existing fieldmap: {fieldmap_rads}")
    else:
        run([
            "fsl_prepare_fieldmap",
            "SIEMENS",
            str(phasediff),
            str(magnitude_brain),
            str(fieldmap_rads),
            str(delta_te_ms),
        ])

    print(f"\nFieldmap preparation complete: {subject}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    args = parser.parse_args()

    prepare_fieldmap(args.subject)
