import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n>", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python scripts/test_robust_epi_registration.py sub-XXXXX"
    )

subject = sys.argv[1]

root = Path("derivatives") / subject

epi = root / "func" / f"{subject}_task-heart_desc-stcMC_mean_bold.nii.gz"
t1brain = root / "anat" / f"{subject}_T1w_brain.nii.gz"

fmap = root / "fmap" / f"{subject}_fieldmap_rads.nii.gz"
mag = root / "fmap" / f"{subject}_magnitude1_brain.nii.gz"
magmask = root / "fmap" / f"{subject}_magnitude1_brain_mask.nii.gz"

wmseg = (
    root
    / "reg"
    / "nofmap"
    / f"{subject}_stcMC_epi2t1_nofmap_fast_wmseg.nii.gz"
)

out = root / "reg" / "diagnostic"
out.mkdir(parents=True, exist_ok=True)

rigid_mat = out / f"{subject}_epi2t1_rigid6.mat"
rigid_img = out / f"{subject}_epi2t1_rigid6.nii.gz"

bbr_mat = out / f"{subject}_epi2t1_rigid6_bbr.mat"
bbr_img = out / f"{subject}_epi2t1_rigid6_bbr.nii.gz"

mag_mat = out / f"{subject}_fmapmag2t1.mat"
mag_img = out / f"{subject}_fmapmag2t1.nii.gz"

fmap_t1 = out / f"{subject}_fieldmap_rads_space-T1.nii.gz"
mask_t1 = out / f"{subject}_fieldmap_mask_space-T1.nii.gz"

final_mat = out / f"{subject}_epi2t1_rigid6_bbr_fmap.mat"
final_img = out / f"{subject}_epi2t1_rigid6_bbr_fmap.nii.gz"


# 1. Wide-angle rigid EPI -> T1
run([
    "flirt",
    "-in", epi,
    "-ref", t1brain,
    "-dof", "6",
    "-cost", "normmi",
    "-searchrx", "-180", "180",
    "-searchry", "-180", "180",
    "-searchrz", "-180", "180",
    "-omat", rigid_mat,
    "-out", rigid_img,
])


# 2. BBR refinement
run([
    "flirt",
    "-in", epi,
    "-ref", t1brain,
    "-init", rigid_mat,
    "-dof", "6",
    "-cost", "bbr",
    "-wmseg", wmseg,
    "-omat", bbr_mat,
    "-out", bbr_img,
])


# 3. Fieldmap magnitude -> T1
run([
    "flirt",
    "-in", mag,
    "-ref", t1brain,
    "-dof", "6",
    "-cost", "normmi",
    "-searchrx", "-180", "180",
    "-searchry", "-180", "180",
    "-searchrz", "-180", "180",
    "-omat", mag_mat,
    "-out", mag_img,
])


# 4. Fieldmap rad/s -> T1
run([
    "flirt",
    "-in", fmap,
    "-ref", t1brain,
    "-applyxfm",
    "-init", mag_mat,
    "-interp", "trilinear",
    "-out", fmap_t1,
])


# 5. Fieldmap mask -> T1
run([
    "flirt",
    "-in", magmask,
    "-ref", t1brain,
    "-applyxfm",
    "-init", mag_mat,
    "-interp", "nearestneighbour",
    "-out", mask_t1,
])


# 6. BBR + fieldmap using FSL BBR schedule
run([
    "flirt",
    "-in", epi,
    "-ref", t1brain,
    "-init", bbr_mat,
    "-wmseg", wmseg,
    "-fieldmap", fmap_t1,
    "-fieldmapmask", mask_t1,
    "-pedir", "-2",
    "-echospacing", "0.000510004",
    "-schedule", str(Path.home() / "fsl/etc/flirtsch/bbr.sch"),
    "-omat", final_mat,
    "-out", final_img,
])


print("\nDiagnostic registration completed.")
print("Final image:", final_img)
print("Final matrix:", final_mat)