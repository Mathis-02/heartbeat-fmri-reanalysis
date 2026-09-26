import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Compare GLM spatial coverage across subjects."
    )
    parser.add_argument(
        "subjects",
        nargs="+",
        help="Subjects to compare, e.g. sub-09113 sub-09210 sub-09260",
    )
    args = parser.parse_args()

    subjects = args.subjects

    output_dir = Path("derivatives/group/qc")
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    reference_img = None

    print("Subjects:")
    for subject in subjects:
        mask_path = (
            Path("derivatives")
            / subject
            / "qc"
            / "glm_mask_final_subject_stcMC.nii.gz"
        )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"GLM mask not found for {subject}: {mask_path}"
            )

        img = nib.load(mask_path)
        data = np.asarray(img.dataobj) > 0

        if reference_img is None:
            reference_img = img
            reference_shape = img.shape
            reference_affine = img.affine
        else:
            if img.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch for {subject}: "
                    f"{img.shape} vs {reference_shape}"
                )

            if not np.allclose(img.affine, reference_affine):
                raise ValueError(
                    f"Affine mismatch for {subject}"
                )

        masks.append(data)

        print(
            f"  {subject}: "
            f"{np.count_nonzero(data)} voxels"
        )

    stack = np.stack(masks, axis=0)

    coverage_count = np.sum(stack, axis=0).astype(np.uint8)
    common_mask = np.all(stack, axis=0).astype(np.uint8)
    union_mask = np.any(stack, axis=0).astype(np.uint8)

    n_subjects = len(subjects)

    coverage_path = output_dir / "coverage_count.nii.gz"
    common_path = output_dir / "common_mask.nii.gz"
    union_path = output_dir / "union_mask.nii.gz"

    nib.save(
        nib.Nifti1Image(
            coverage_count,
            reference_affine,
            reference_img.header,
        ),
        coverage_path,
    )

    nib.save(
        nib.Nifti1Image(
            common_mask,
            reference_affine,
            reference_img.header,
        ),
        common_path,
    )

    nib.save(
        nib.Nifti1Image(
            union_mask,
            reference_affine,
            reference_img.header,
        ),
        union_path,
    )

    print("\nCoverage distribution:")

    for n in range(n_subjects + 1):
        count = np.count_nonzero(coverage_count == n)
        print(
            f"  Covered by {n}/{n_subjects} subjects: "
            f"{count} voxels"
        )

    common_voxels = np.count_nonzero(common_mask)
    union_voxels = np.count_nonzero(union_mask)

    print(f"\nUnion voxels:  {union_voxels}")
    print(f"Common voxels: {common_voxels}")

    if union_voxels > 0:
        print(
            "Common / union: "
            f"{100 * common_voxels / union_voxels:.2f}%"
        )

    print("\nPairwise Dice coefficients:")

    for i in range(n_subjects):
        for j in range(i + 1, n_subjects):
            a = masks[i]
            b = masks[j]

            intersection = np.count_nonzero(a & b)
            denominator = (
                np.count_nonzero(a)
                + np.count_nonzero(b)
            )

            dice = (
                2 * intersection / denominator
                if denominator > 0
                else np.nan
            )

            print(
                f"  {subjects[i]} vs {subjects[j]}: "
                f"{dice:.4f}"
            )

    print("\nSaved:")
    print(f"  {coverage_path}")
    print(f"  {common_path}")
    print(f"  {union_path}")


if __name__ == "__main__":
    main()