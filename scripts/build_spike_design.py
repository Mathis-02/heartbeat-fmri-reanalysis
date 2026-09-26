from pathlib import Path
import sys

import numpy as np
import pandas as pd


FD_THRESHOLD = 0.5


def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/build_spike_design.py sub-XXXXX"
        )

    subject = sys.argv[1]

    qc_dir = Path("derivatives") / subject / "qc"

    design_path = qc_dir / "design_matrix_stcMC.tsv"
    fd_path = qc_dir / "stcMC" / "framewise_displacement.tsv"

    output_path = qc_dir / "design_matrix_stcMC_spikeFD05.tsv"

    design = pd.read_csv(design_path, sep="\t")
    fd = pd.read_csv(fd_path, sep="\t")["FD_mm"].to_numpy()

    if len(design) != len(fd):
        raise ValueError(
            f"Length mismatch: design={len(design)}, FD={len(fd)}"
        )

    spike_volumes = np.where(fd > FD_THRESHOLD)[0]

    print(f"Subject: {subject}")
    print(f"Volumes: {len(fd)}")
    print(f"FD threshold: > {FD_THRESHOLD} mm")
    print(f"Spike volumes: {len(spike_volumes)}")
    print(f"Indices: {spike_volumes.tolist()}")

    spike_regressors = {}

    for volume in spike_volumes:

        regressor = np.zeros(len(fd))
        regressor[volume] = 1.0

        spike_regressors[f"spike_{volume:03d}"] = regressor

    if spike_regressors:
        spikes = pd.DataFrame(spike_regressors)
        design_spike = pd.concat(
            [design, spikes],
            axis=1,
        )
    else:
        design_spike = design.copy()

    matrix = design_spike.to_numpy(dtype=float)

    rank = np.linalg.matrix_rank(matrix)

    print()
    print(f"Original design: {design.shape}")
    print(f"Spike design:    {design_spike.shape}")
    print(f"Matrix rank:     {rank}/{design_spike.shape[1]}")

    if rank != design_spike.shape[1]:
        raise RuntimeError(
            "Design matrix is not full rank."
        )

    design_spike.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()