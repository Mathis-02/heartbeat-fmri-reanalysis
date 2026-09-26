from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from itertools import product

output_dir = Path("results/mvpa")
output_dir.mkdir(parents=True, exist_ok=True)

summary_results = []
null_results = []


subjects = [
    "sub-09113", "sub-09210", "sub-09260",
    "sub-09261", "sub-09301", "sub-09380",
    "sub-09548", "sub-09587", "sub-09607",
]

tr = 2.0
hrf_delay = 4.0
n_features = 1000


for sub in subjects:
    bold_file = Path(
        f"derivatives/{sub}/func/"
        f"{sub}_task-heart_desc-stcMC_bold.nii.gz"
    )

    events_file = Path(
        f"data/ds003763/{sub}/func/"
        f"{sub}_task-heart_events.tsv"
    )

    mask_file = Path(
        f"derivatives/{sub}/qc/native/"
        f"bold_brain_mask_stcMC.nii.gz"
    )

    img = nib.load(bold_file)
    bold = np.asarray(img.dataobj)

    events = pd.read_csv(events_file, sep="\t")

    patterns = []
    labels = []

    for _, event in events.iterrows():
        start = int((event["onset"] + hrf_delay) / tr)
        stop = int(
            (event["onset"] + event["duration"] + hrf_delay) / tr
        )

        block = bold[..., start:stop]
        patterns.append(block.mean(axis=3))
        labels.append(event["trial_type"])

    patterns = np.stack(patterns)

    mask = nib.load(mask_file).get_fdata() > 0

    if mask.shape != patterns.shape[1:]:
        raise ValueError(
            f"{sub}: mask {mask.shape} "
            f"!= BOLD {patterns.shape[1:]}"
        )

    X = patterns[:, mask]
    y = np.array(labels)

    # Each group contains one Sound block and the following Heart block.
    groups = np.repeat(np.arange(8), 2)

    model = make_pipeline(
        SelectKBest(f_classif, k=n_features),
        StandardScaler(),
        LinearSVC(
            C=1.0,
            dual="auto",
            max_iter=10000,
        ),
    )

    cv = LeaveOneGroupOut()

    scores = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="accuracy",
    )
    observed_accuracy = scores.mean()
    permutation_accuracies = []

    for swaps in product([0, 1], repeat=8):
        y_perm = y.copy()

        for pair, swap in enumerate(swaps):
            if swap:
                i = 2 * pair
                y_perm[i], y_perm[i + 1] = (
                    y_perm[i + 1],
                    y_perm[i],
                )

        perm_scores = cross_val_score(
            model,
            X,
            y_perm,
            groups=groups,
            cv=cv,
            scoring="accuracy",
        )

        permutation_accuracies.append(
            perm_scores.mean()
        )

    permutation_accuracies = np.array(
        permutation_accuracies
    )

    exact_p = np.mean(
        permutation_accuracies >= observed_accuracy
    )

    print(
        sub,
        "| accuracy:",
        round(float(observed_accuracy), 3),
        "| exact p:",
        round(float(exact_p), 4),
    )

    summary_results.append({
        "subject": sub,
        "accuracy": observed_accuracy,
        "p_value": exact_p,
    })

    for permutation, accuracy in enumerate(permutation_accuracies):
        null_results.append({
            "subject": sub,
            "permutation": permutation,
            "accuracy": accuracy,
        })

summary_df = pd.DataFrame(summary_results)
null_df = pd.DataFrame(null_results)

summary_df.to_csv(
    output_dir / "mvpa_subject_results.csv",
    index=False,
)

null_df.to_csv(
    output_dir / "mvpa_permutation_accuracies.csv",
    index=False,
)

print("\nMean accuracy:", round(summary_df["accuracy"].mean(), 3))
print("Results saved in", output_dir)