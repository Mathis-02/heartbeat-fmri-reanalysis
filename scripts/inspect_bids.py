from pathlib import Path
import argparse
import json

import nibabel as nib
import numpy as np


DATASET = Path("data/ds003763")


def inspect_subject(subject: str) -> dict:
    """Validate the inputs required for one subject's heart-task run."""

    subject_dir = DATASET / subject

    bold = (
        subject_dir / "func"
        / f"{subject}_task-heart_bold.nii.gz"
    )
    bold_json = (
        subject_dir / "func"
        / f"{subject}_task-heart_bold.json"
    )
    events = (
        subject_dir / "func"
        / f"{subject}_task-heart_events.tsv"
    )
    t1w = (
        subject_dir / "anat"
        / f"{subject}_T1w.nii.gz"
    )

    for path in (bold, bold_json, events, t1w):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    metadata = json.loads(bold_json.read_text())
    image = nib.load(bold)

    if len(image.shape) != 4:
        raise ValueError(f"Expected 4D BOLD: {image.shape}")

    n_slices = image.shape[2]
    n_volumes = image.shape[3]

    tr = float(metadata["RepetitionTime"])
    slice_timing = np.asarray(
        metadata["SliceTiming"], dtype=float
    )

    if tr <= 0 or not np.isfinite(tr):
        raise ValueError(f"Invalid TR: {tr}")

    if (
        len(slice_timing) != n_slices
        or not np.all(np.isfinite(slice_timing))
        or np.any(slice_timing < 0)
        or np.any(slice_timing >= tr)
    ):
        raise ValueError("Invalid SliceTiming metadata")

    phase_encoding = metadata.get("PhaseEncodingDirection")
    echo_spacing = metadata.get("EffectiveEchoSpacing")

    fmap_dir = subject_dir / "fmap"
    associated_fieldmaps = []

    for fmap_json in sorted(fmap_dir.glob("*_phasediff.json")):
        fmap_metadata = json.loads(fmap_json.read_text())
        intended_for = fmap_metadata.get("IntendedFor", [])

        if isinstance(intended_for, str):
            intended_for = [intended_for]

        target = f"func/{bold.name}"

        if any(
            str(item).removeprefix("bids::").lstrip("/") == target
            for item in intended_for
        ):
            phasediff = Path(
                str(fmap_json).removesuffix(".json") + ".nii.gz"
            )
            magnitude = fmap_dir / (
                fmap_json.name.replace(
                    "_phasediff.json", "_magnitude1.nii.gz"
                )
            )

            if not phasediff.is_file() or not magnitude.is_file():
                raise FileNotFoundError(
                    f"Incomplete associated fieldmap: {fmap_json}"
                )

            te1 = float(fmap_metadata["EchoTime1"])
            te2 = float(fmap_metadata["EchoTime2"])

            if te2 <= te1:
                raise ValueError("Invalid fieldmap echo times")

            associated_fieldmaps.append({
                "phasediff": str(phasediff),
                "magnitude1": str(magnitude),
                "delta_te": te2 - te1,
            })

    if len(associated_fieldmaps) > 1:
        raise ValueError(
            "Multiple associated fieldmaps: manual selection required"
        )

    fieldmap = (
        associated_fieldmaps[0]
        if associated_fieldmaps else None
    )

    return {
        "subject": subject,
        "bold": str(bold),
        "events": str(events),
        "t1w": str(t1w),
        "shape": list(image.shape),
        "n_slices": n_slices,
        "n_volumes": n_volumes,
        "tr": tr,
        "slice_timing": slice_timing.tolist(),
        "phase_encoding": phase_encoding,
        "effective_echo_spacing": echo_spacing,
        "fieldmap": fieldmap,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    args = parser.parse_args()

    result = inspect_subject(args.subject)
    print(json.dumps(result, indent=2))
