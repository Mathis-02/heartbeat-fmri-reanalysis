from pathlib import Path
import argparse

import nibabel as nib
import numpy as np


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return np.asarray(
        nib.load(path).dataobj,
        dtype=np.float64,
    )


def describe(name, z, effect, t):
    positive = z >= 3
    negative = z <= -3
    absolute = np.abs(z) >= 3

    valid_se = np.abs(t) > 1e-8
    se = np.abs(effect[valid_se]) / np.abs(t[valid_se])

    print(f"\n{name}")
    print("-" * 50)

    print(f"Z mean: {np.mean(z):.6f}")
    print(f"Z SD:   {np.std(z):.6f}")

    print(
        f"Z >= 3:  {positive.sum()} "
        f"({100 * positive.mean():.6f}%)"
    )
    print(
        f"Z <= -3: {negative.sum()} "
        f"({100 * negative.mean():.6f}%)"
    )
    print(
        f"|Z| >= 3: {absolute.sum()} "
        f"({100 * absolute.mean():.6f}%)"
    )

    print(f"Effect mean:   {np.mean(effect):.6f}")
    print(f"Effect SD:     {np.std(effect):.6f}")
    print(f"Effect median: {np.median(effect):.6f}")

    print(f"SE mean:   {np.mean(se):.6f}")
    print(f"SE median: {np.median(se):.6f}")


def main(subject):

    derivatives = Path("derivatives") / subject

    original_dir = derivatives / "stats" / "stcMC"
    spike_dir = derivatives / "stats" / "stcMC_spikeFD05"

    mask_path = (
        derivatives
        / "qc"
        / "glm_mask_final_subject_stcMC.nii.gz"
    )

    mask = load(mask_path) > 0

    original_z = load(
        original_dir / "heart_minus_sound_z.nii.gz"
    )[mask]

    original_effect = load(
        original_dir / "heart_minus_sound_effect.nii.gz"
    )[mask]

    original_t = load(
        original_dir / "heart_minus_sound_t.nii.gz"
    )[mask]

    spike_z = load(
        spike_dir / "heart_minus_sound_z.nii.gz"
    )[mask]

    spike_effect = load(
        spike_dir / "heart_minus_sound_effect.nii.gz"
    )[mask]

    spike_t = load(
        spike_dir / "heart_minus_sound_t.nii.gz"
    )[mask]

    finite = (
        np.isfinite(original_z)
        & np.isfinite(original_effect)
        & np.isfinite(original_t)
        & np.isfinite(spike_z)
        & np.isfinite(spike_effect)
        & np.isfinite(spike_t)
    )

    original_z = original_z[finite]
    original_effect = original_effect[finite]
    original_t = original_t[finite]

    spike_z = spike_z[finite]
    spike_effect = spike_effect[finite]
    spike_t = spike_t[finite]

    print(f"Subject: {subject}")
    print(f"Compared voxels: {len(original_z)}")

    describe(
        "ORIGINAL",
        original_z,
        original_effect,
        original_t,
    )

    describe(
        "SPIKE FD > 0.5",
        spike_z,
        spike_effect,
        spike_t,
    )

    print("\n" + "=" * 50)
    print("MAP COMPARISON")
    print("=" * 50)

    print(
        f"Z correlation: "
        f"{np.corrcoef(original_z, spike_z)[0, 1]:.6f}"
    )

    print(
        f"Effect correlation: "
        f"{np.corrcoef(original_effect, spike_effect)[0, 1]:.6f}"
    )

    z_diff = spike_z - original_z
    effect_diff = spike_effect - original_effect

    print(
        f"Mean Z difference (spike - original): "
        f"{np.mean(z_diff):.6f}"
    )

    print(
        f"Mean |Z difference|: "
        f"{np.mean(np.abs(z_diff)):.6f}"
    )

    print(
        f"Z RMSE: "
        f"{np.sqrt(np.mean(z_diff ** 2)):.6f}"
    )

    print(
        f"Mean effect difference: "
        f"{np.mean(effect_diff):.6f}"
    )

    print(
        f"Mean |effect difference|: "
        f"{np.mean(np.abs(effect_diff)):.6f}"
    )

    print(
        f"Effect RMSE: "
        f"{np.sqrt(np.mean(effect_diff ** 2)):.6f}"
    )

    original_class = np.zeros(len(original_z), dtype=np.int8)
    original_class[original_z >= 3] = 1
    original_class[original_z <= -3] = -1

    spike_class = np.zeros(len(spike_z), dtype=np.int8)
    spike_class[spike_z >= 3] = 1
    spike_class[spike_z <= -3] = -1

    changed = original_class != spike_class

    print(
        f"Threshold classification changed: "
        f"{changed.sum()} "
        f"({100 * changed.mean():.6f}%)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    args = parser.parse_args()

    main(args.subject)