from pathlib import Path

import numpy as np
import pandas as pd


results_dir = Path("results/mvpa")

summary = pd.read_csv(
    results_dir / "mvpa_subject_results.csv"
)

null = pd.read_csv(
    results_dir / "mvpa_permutation_accuracies.csv"
)

observed_mean = summary["accuracy"].mean()

subjects = summary["subject"].tolist()

null_by_subject = {
    sub: null.loc[
        null["subject"] == sub, "accuracy"
    ].to_numpy()
    for sub in subjects
}

rng = np.random.default_rng(42)

n_group_permutations = 100000
group_null = np.empty(n_group_permutations)

for i in range(n_group_permutations):
    sampled = [
        rng.choice(null_by_subject[sub])
        for sub in subjects
    ]

    group_null[i] = np.mean(sampled)

p_value = (
    np.sum(group_null >= observed_mean) + 1
) / (n_group_permutations + 1)

print("Subjects:", len(subjects))
print("Observed mean accuracy:", round(observed_mean, 4))
print("Chance level: 0.5")
print("Null mean:", round(group_null.mean(), 4))
print("Group permutation p:", round(p_value, 5))

np.save(
    results_dir / "mvpa_group_null.npy",
    group_null,
)