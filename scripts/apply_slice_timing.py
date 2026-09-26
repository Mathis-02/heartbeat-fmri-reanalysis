from pathlib import Path
import json
import subprocess
import sys

import nibabel as nib
import numpy as np


# --- Subject from command line ---

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python scripts/apply_slice_timing.py sub-XXXXX"
    )

SUBJECT = sys.argv[1]


# --- Input files ---

BOLD_FILE = Path(
    f"data/ds003763/{SUBJECT}/func/"
    f"{SUBJECT}_task-heart_bold.nii.gz"
)

JSON_FILE = Path(
    f"data/ds003763/{SUBJECT}/func/"
    f"{SUBJECT}_task-heart_bold.json"
)

OUTPUT_DIR = Path(
    f"derivatives/{SUBJECT}/func"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STC_FILE = OUTPUT_DIR / (
    f"{SUBJECT}_task-heart_desc-stc_bold.nii.gz"
)

ORDER_FILE = OUTPUT_DIR / (
    f"{SUBJECT}_slice_order.txt"
)


# --- Check files ---

if not BOLD_FILE.exists():
    raise FileNotFoundError(BOLD_FILE)

if not JSON_FILE.exists():
    raise FileNotFoundError(JSON_FILE)


# --- Read metadata ---

with open(JSON_FILE, "r") as f:
    metadata = json.load(f)

if "SliceTiming" not in metadata:
    raise ValueError(
        f"No SliceTiming field in {JSON_FILE}"
    )

slice_timing = np.asarray(
    metadata["SliceTiming"],
    dtype=float,
)

tr = float(
    metadata["RepetitionTime"]
)


# --- Check BOLD dimensions ---

bold_img = nib.load(BOLD_FILE)

n_slices = bold_img.shape[2]

if len(slice_timing) != n_slices:
    raise ValueError(
        f"SliceTiming contains {len(slice_timing)} values "
        f"but BOLD has {n_slices} slices."
    )


# --- Check for simultaneous/multiband acquisition ---

unique_times = np.unique(slice_timing)

if len(unique_times) != n_slices:
    raise ValueError(
        "Repeated SliceTiming values detected. "
        "This may be a multiband acquisition. "
        "Do not generate a simple slice-order file."
    )


# --- Determine acquisition order ---

# np.argsort returns 0-based NIfTI slice indices
order_zero_based = np.argsort(slice_timing)

# FSL --ocustom expects slices numbered from 1
order_fsl = order_zero_based + 1


# --- Save FSL slice order file ---

np.savetxt(
    ORDER_FILE,
    order_fsl,
    fmt="%d",
)


# --- Print QC information ---

print(f"Subject: {SUBJECT}")
print(f"TR: {tr:.3f} s")
print(f"Number of slices: {n_slices}")

print("\nSlice timing range:")
print(
    f"{slice_timing.min():.6f} "
    f"to {slice_timing.max():.6f} s"
)

print("\nFSL slice acquisition order:")
print(order_fsl)

print(f"\nOrder file: {ORDER_FILE}")


# --- Run FSL slicetimer ---

command = [
    "slicetimer",
    "-i", str(BOLD_FILE),
    "-o", str(STC_FILE),
    "-r", str(tr),
    f"--ocustom={ORDER_FILE}",
]

print("\nRunning:")
print(" ".join(command))

subprocess.run(
    command,
    check=True,
)

print(f"\nCreated: {STC_FILE}")