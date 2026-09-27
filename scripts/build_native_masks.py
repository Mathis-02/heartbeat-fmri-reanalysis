from pathlib import Path
import argparse

from nilearn.masking import compute_epi_mask


def build_native_mask(subject):
    bold = (
        Path("derivatives")
        / subject
        / "func"
        / f"{subject}_task-heart_desc-stcMC_bold.nii.gz"
    )

    if not bold.is_file():
        raise FileNotFoundError(bold)

    out_dir = Path("derivatives") / subject / "qc" / "native"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = out_dir / "bold_brain_mask_stcMC.nii.gz"

    mask = compute_epi_mask(str(bold))
    mask.to_filename(output)

    print(f"Native functional mask saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    args = parser.parse_args()

    build_native_mask(args.subject)
