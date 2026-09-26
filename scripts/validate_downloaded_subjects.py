from pathlib import Path
import json

import nibabel as nib


SUBJECTS = [
    "sub-09261",
    "sub-09301",
    "sub-09380",
    "sub-09381",
    "sub-09548",
    "sub-09587",
    "sub-09607",
]

ROOT = Path("sourcedata/ds003763")


def main():

    for subject in SUBJECTS:

        subject_dir = ROOT / subject

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

        print("=" * 70)
        print(subject)

        required = [bold, bold_json, events, t1w]

        missing = [
            str(path)
            for path in required
            if not path.is_file()
        ]

        if missing:
            print("FAILED: missing required files")
            for path in missing:
                print(" ", path)
            continue

        try:
            img = nib.load(bold)

            if img.ndim != 4:
                raise ValueError(
                    f"BOLD is not 4D: {img.shape}"
                )

            # Force access to the final volume.
            # This checks that NiBabel can actually read
            # data near the end of the image.
            _ = img.dataobj[..., -1]

            metadata = json.loads(
                bold_json.read_text(
                    encoding="utf-8"
                )
            )

            tr_header = float(
                img.header.get_zooms()[3]
            )

            tr_json = metadata.get(
                "RepetitionTime"
            )

            slice_timing = metadata.get(
                "SliceTiming"
            )

            phase_encoding = metadata.get(
                "PhaseEncodingDirection"
            )

            effective_echo_spacing = metadata.get(
                "EffectiveEchoSpacing"
            )

            fmap_dir = subject_dir / "fmap"

            phasediff = (
                fmap_dir
                / f"{subject}_phasediff.nii.gz"
            )

            magnitude1 = (
                fmap_dir
                / f"{subject}_magnitude1.nii.gz"
            )

            fieldmap = (
                phasediff.is_file()
                and magnitude1.is_file()
            )

            print(f"BOLD shape:        {img.shape}")
            print(f"TR header:         {tr_header}")
            print(f"TR JSON:           {tr_json}")
            print(
                "Slice timings:     "
                f"{len(slice_timing) if slice_timing else None}"
            )
            print(
                f"Phase encoding:    {phase_encoding}"
            )
            print(
                "Effective echo:    "
                f"{effective_echo_spacing}"
            )
            print(f"Events:            OK")
            print(f"T1w:               OK")
            print(
                "Fieldmap files:    "
                f"{'YES' if fieldmap else 'NO'}"
            )
            print("NIfTI read:        OK")

        except Exception as exc:
            print(
                f"FAILED: {type(exc).__name__}: {exc}"
            )

        print()


if __name__ == "__main__":
    main()