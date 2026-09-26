from pathlib import Path
import nibabel as nib
import numpy as np
import json

SUBJECTS = "sub-09113"
DATASET = Path("data/ds003763")

BOLD_FILE = (
    DATASET
    / SUBJECTS
    / "func"
    / f"{SUBJECTS}_task-heart_bold.nii.gz"
)

JSON_FILE = (
    DATASET
    / SUBJECTS
    / "func"
    / f"{SUBJECTS}_task-heart_bold.json"
)

bold = nib.load(BOLD_FILE)

with open(JSON_FILE, "r") as f:
    metadata = json.load(f)

tr = float(metadata["RepetitionTime"])
slice_times = np.asarray(metadata["SliceTiming"], dtype=float)

n_slices = bold.shape[2]

if len(slice_times) != n_slices:
    raise ValueError(
        f"{len(slice_times)} SliceTiming values "
        f"but BOLD has {n_slices} slices."
    )

# SliceTiming[i] = acquisition time of slice i
order = np.argsort(slice_times)

reference_time = tr / 2
time_shifts = reference_time - slice_times

print(f"TR: {tr:.3f} s")
print(f"Number of slices: {n_slices}")
print(f"Reference time: {reference_time:.3f} s")

print("\nAcquisition order (0-based NIfTI indices):")
print(order.tolist())

print("\nSlice | acquisition time | shift to middle of TR")

for slice_index in order:
    print(
        f"{slice_index:5d} | "
        f"{slice_times[slice_index]:8.4f} s | "
        f"{time_shifts[slice_index]:+8.4f} s"
    )

output_dir = Path("derivatives/work") / SUBJECTS
output_dir.mkdir(parents=True, exist_ok=True)

fsl_order = order + 1

order_file = output_dir / "slice_order.txt"

np.savetxt(
    order_file,
    fsl_order,
    fmt="%d"
)

print(f"\nFSL slice order written to: {order_file}")