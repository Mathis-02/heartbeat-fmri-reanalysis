from pathlib import Path

import nibabel as nib
import numpy as np


SUBJECTS = [
    "sub-09113",
    "sub-09210",
    "sub-09260",
]

COMMON_MASK = Path(
    "derivatives/group/qc/common_mask.nii.gz"
)


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return np.asarray(
        nib.load(path).dataobj,
        dtype=np.float64,
    )


def main():

    common = load_data(COMMON_MASK) > 0
    n_common = np.count_nonzero(common)

    print(f"Common mask voxels: {n_common}")
    print()

    for subject in SUBJECTS:

        stats_dir = (
            Path("derivatives")
            / subject
            / "stats"
            / "stcMC"
        )

        z_path = stats_dir / "heart_minus_sound_z.nii.gz"
        effect_path = (
            stats_dir
            / "heart_minus_sound_effect.nii.gz"
        )
        t_path = stats_dir / "heart_minus_sound_t.nii.gz"

        z = load_data(z_path)[common]
        effect = load_data(effect_path)[common]
        t = load_data(t_path)[common]

        finite = (
            np.isfinite(z)
            & np.isfinite(effect)
            & np.isfinite(t)
        )

        z = z[finite]
        effect = effect[finite]
        t = t[finite]

        positive = z >= 3
        negative = z <= -3
        suprathreshold = np.abs(z) >= 3

        valid_se = np.abs(t) > 1e-8
        se = (
            np.abs(effect[valid_se])
            / np.abs(t[valid_se])
        )

        print("=" * 60)
        print(subject)
        print("=" * 60)

        print(f"Finite voxels: {len(z)}")

        print("\nZ distribution:")
        print(f"  mean:   {np.mean(z):.6f}")
        print(f"  sd:     {np.std(z):.6f}")
        print(f"  min:    {np.min(z):.6f}")
        print(f"  max:    {np.max(z):.6f}")

        print("\nDescriptive |Z| >= 3:")
        print(
            f"  positive: {np.count_nonzero(positive)} "
            f"({100*np.mean(positive):.6f}%)"
        )
        print(
            f"  negative: {np.count_nonzero(negative)} "
            f"({100*np.mean(negative):.6f}%)"
        )
        print(
            f"  absolute: {np.count_nonzero(suprathreshold)} "
            f"({100*np.mean(suprathreshold):.6f}%)"
        )

        print("\nEffect distribution:")
        print(f"  mean:   {np.mean(effect):.6f}")
        print(f"  sd:     {np.std(effect):.6f}")
        print(f"  median: {np.median(effect):.6f}")
        print(
            f"  p05:    {np.percentile(effect, 5):.6f}"
        )
        print(
            f"  p95:    {np.percentile(effect, 95):.6f}"
        )

        print("\nContrast SE:")
        print(f"  mean:   {np.mean(se):.6f}")
        print(f"  median: {np.median(se):.6f}")
        print(f"  sd:     {np.std(se):.6f}")
        print(
            f"  p05:    {np.percentile(se, 5):.6f}"
        )
        print(
            f"  p95:    {np.percentile(se, 95):.6f}"
        )

        print()


if __name__ == "__main__":
    main()