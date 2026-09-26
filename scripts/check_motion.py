from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# --- Subject from command line ---

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python scripts/check_motion.py sub-XXXXX"
    )

SUBJECT = sys.argv[1]


# --- Files ---

PAR_FILE = Path(
    f"derivatives/{SUBJECT}/func/"
    f"{SUBJECT}_task-heart_desc-stcMC_bold.par"
)

QC_DIR = Path(f"derivatives/{SUBJECT}/qc/stcMC")

QC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if not PAR_FILE.exists():
    raise FileNotFoundError(
        f"Motion file not found: {PAR_FILE}"
    )


# --- Load MCFLIRT parameters ---

motion = np.loadtxt(PAR_FILE)

if motion.ndim != 2 or motion.shape[1] != 6:
    raise ValueError(
        f"Expected 6 MCFLIRT parameters, got shape {motion.shape}"
    )


# MCFLIRT .par convention:
# columns 0-2 = rotations in radians
# columns 3-5 = translations in mm

rotations = motion[:, :3]
translations = motion[:, 3:]


# --- Framewise displacement: Power FD ---

# Difference between successive volumes
delta_rot = np.diff(
    rotations,
    axis=0,
)

delta_trans = np.diff(
    translations,
    axis=0,
)

# Convert rotations to approximate displacement
# on a sphere of radius 50 mm
HEAD_RADIUS_MM = 50.0

fd = (
    np.abs(delta_trans).sum(axis=1)
    +
    HEAD_RADIUS_MM
    * np.abs(delta_rot).sum(axis=1)
)

# First volume has no previous volume to compare with
fd = np.insert(
    fd,
    0,
    0.0,
)


# --- Summary statistics ---

FD_THRESHOLD = 0.5

n_volumes = len(fd)

n_high_fd = np.sum(
    fd > FD_THRESHOLD
)

pct_high_fd = (
    100 * n_high_fd / n_volumes
)

print(f"Subject: {SUBJECT}")
print(f"Volumes: {n_volumes}")
print(f"Mean FD: {fd.mean():.3f} mm")
print(f"Median FD: {np.median(fd):.3f} mm")
print(f"Max FD: {fd.max():.3f} mm")
print(
    f"Volumes FD > {FD_THRESHOLD} mm: "
    f"{n_high_fd}/{n_volumes} "
    f"({pct_high_fd:.2f}%)"
)


# --- Save FD values ---

np.savetxt(
    QC_DIR / "framewise_displacement.tsv",
    fd,
    delimiter="\t",
    header="FD_mm",
    comments="",
)


# --- Plot FD ---

fig, ax = plt.subplots(
    figsize=(12, 4)
)

ax.plot(fd)

ax.axhline(
    FD_THRESHOLD,
    linestyle="--",
    label=f"{FD_THRESHOLD} mm threshold",
)

ax.set_xlabel("Volume")
ax.set_ylabel("Framewise displacement (mm)")
ax.set_title(
    f"{SUBJECT} — Power FD"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    QC_DIR / "framewise_displacement.png",
    dpi=150,
)

plt.close()


# --- Plot MCFLIRT parameters ---

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 7),
    sharex=True,
)

axes[0].plot(
    translations
)

axes[0].set_ylabel(
    "Translation (mm)"
)

axes[0].set_title(
    f"{SUBJECT} — MCFLIRT translations"
)

axes[0].legend(
    ["x", "y", "z"]
)

axes[1].plot(
    rotations
)

axes[1].set_ylabel(
    "Rotation (rad)"
)

axes[1].set_xlabel(
    "Volume"
)

axes[1].set_title(
    f"{SUBJECT} — MCFLIRT rotations"
)

axes[1].legend(
    ["x", "y", "z"]
)

plt.tight_layout()

plt.savefig(
    QC_DIR / "motion_parameters.png",
    dpi=150,
)

plt.close()