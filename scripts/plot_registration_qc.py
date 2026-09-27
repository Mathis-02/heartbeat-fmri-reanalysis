from pathlib import Path
import os

from nilearn import plotting


subject = "sub-09113"

t1 = Path(
    f"data/ds003763/{subject}/anat/"
    f"{subject}_T1w.nii.gz"
)

epi_t1 = Path(
    f"derivatives/{subject}/reg/"
    f"{subject}_stcMC_epi2t1.nii.gz"
)

t1_mni = Path(
    f"derivatives/{subject}/reg/ants/final/"
    f"{subject}_T1w_space-MNI.nii.gz"
)

mni = (
    Path(os.environ["FSLDIR"])
    / "data/standard/MNI152_T1_2mm.nii.gz"
)

output_dir = Path("results/registration")
output_dir.mkdir(parents=True, exist_ok=True)


# EPI registered to anatomical T1
display = plotting.plot_anat(
    t1,
    display_mode="ortho",
    cut_coords=(0, -25, -14),
    title="EPI to T1 registration — sub-09113",
    colorbar=False,
    draw_cross=False,
)

display.add_overlay(
    epi_t1,
    transparency=0.45,
)

display.savefig(
    output_dir / "sub-09113_epi_to_t1_qc.png",
    dpi=180,
)

display.close()


# Anatomical T1 normalized to MNI
display = plotting.plot_anat(
    mni,
    display_mode="ortho",
    cut_coords=(0, -14, 20),
    title="T1 to MNI registration — sub-09113",
    colorbar=False,
    draw_cross=False,
)

display.add_overlay(
    t1_mni,
    transparency=0.45,
)

display.savefig(
    output_dir / "sub-09113_t1_to_mni_qc.png",
    dpi=180,
)

display.close()
