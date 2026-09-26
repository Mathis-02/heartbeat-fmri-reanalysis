from pathlib import Path
import json

import pandas as pd


def main():
    derivatives = Path("derivatives")

    summaries = []

    for subject_dir in sorted(derivatives.glob("sub-*")):

        qc_file = subject_dir / "qc" / "qc_summary.json"

        if not qc_file.is_file():
            continue

        with qc_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        summaries.append({
            "subject": data["subject"],
            "n_volumes": data["n_volumes"],
            "mean_fd_mm": data["motion"]["mean_fd_mm"],
            "median_fd_mm": data["motion"]["median_fd_mm"],
            "max_fd_mm": data["motion"]["max_fd_mm"],
            "n_fd_gt_0.5": data["motion"]["n_fd_above_threshold"],
            "pct_fd_gt_0.5": data["motion"]["pct_fd_above_threshold"],
            "glm_mask_voxels": data["glm_mask_voxels"],
            "design_rows": data["design"]["n_rows"],
            "design_columns": data["design"]["n_columns"],
            "design_rank": data["design"]["rank"],
            "condition_number": (
                data["design"]["condition_number_standardized"]
            ),
        })

    if not summaries:
        raise RuntimeError(
            "No qc_summary.json files found."
        )

    df = pd.DataFrame(summaries)

    output_dir = derivatives / "group" / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "group_qc_summary.tsv"

    df.to_csv(
        output_file,
        sep="\t",
        index=False,
    )

    print("\nGroup QC summary")
    print("=" * 100)

    display_columns = [
        "subject",
        "mean_fd_mm",
        "max_fd_mm",
        "n_fd_gt_0.5",
        "pct_fd_gt_0.5",
        "glm_mask_voxels",
        "condition_number",
    ]

    print(
        df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print(f"Subjects: {len(df)}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()