from pathlib import Path

from nilearn.masking import compute_epi_mask


subjects = [
    "sub-09113", "sub-09210", "sub-09260", "sub-09261",
    "sub-09301", "sub-09380", "sub-09381", "sub-09548",
    "sub-09587", "sub-09607",
]

for subject in subjects:

    bold = (
        Path("derivatives")
        / subject
        / "func"
        / f"{subject}_task-heart_desc-stcMC_bold.nii.gz"
    )

    out_dir = Path("derivatives") / subject / "qc" / "native"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = out_dir / "bold_brain_mask_stcMC.nii.gz"

    print(f"Processing {subject}")

    mask = compute_epi_mask(str(bold))
    mask.to_filename(output)

    print("Saved:", output)
